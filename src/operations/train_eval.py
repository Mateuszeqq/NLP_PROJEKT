from datasets import load_metric
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

sbert = SentenceTransformer('all-mpnet-base-v2')


def tokenize(x):
    return sbert.tokenize(x)


def train_model(model, optimizer, epochs, train_dataloader, criterion, device):
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_dataloader:
            data, labels = batch
            data_batch = tokenize(data)
            labels = torch.tensor(labels, device=device)

            labels = labels.to(device)
            input_ids = data_batch['input_ids'].to(device)
            attention_mask = data_batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f'EPOCH {[epoch]} | LOSS: {np.mean(losses)}')


def get_accuracy(model, test_dataloader, device):
    y_true = []
    y_pred = []

    metric = load_metric("accuracy")
    model.eval()
    for batch in test_dataloader:
        data, labels = batch
        data_batch = tokenize(data)

        labels = torch.tensor(labels, device=device)
        y_true.extend(labels.cpu().numpy())
        labels = labels.to(device)
        input_ids = data_batch['input_ids'].to(device)
        attention_mask = data_batch['attention_mask'].to(device)

        predictions = model(input_ids, attention_mask)
        predictions = torch.argmax(predictions, dim=1)
        y_pred.extend(predictions.cpu().numpy())
        metric.add_batch(predictions=predictions, references=labels)
    return metric.compute()['accuracy'], y_true, y_pred


def train_model_connected(model_connected, optimizer_connected, epochs, train_dataloader, criterion, device):
    model_connected.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_dataloader:
            data, labels = batch
            data_batch = tokenize(data)

            labels = torch.transpose(labels, 0, 1)

            label_1 = labels[0].to(device)
            label_2 = labels[1].to(device)

            input_ids = data_batch['input_ids'].to(device)
            attention_mask = data_batch['attention_mask'].to(device)

            optimizer_connected.zero_grad()
            outputs = model_connected(input_ids, attention_mask)

            output_1 = outputs['class_1']
            output_2 = outputs['class_2']
            output_1.to(device)
            output_2.to(device)

            loss_1 = criterion(output_1, label_1)
            loss_2 = criterion(output_2, label_2)
            loss = loss_1 + loss_2
            loss.backward()
            losses.append(loss.item())
            optimizer_connected.step()
        print(f'EPOCH {[epoch]} | LOSS: {np.mean(losses)}')


def get_accuracy_connected(model_connected, test_dataloader, device):
    model_connected.eval()

    y_true_1 = []
    y_true_2 = []
    y_pred_1 = []
    y_pred_2 = []

    for c in ['class_1', 'class_2']:
        metric= load_metric("accuracy")
        for batch in test_dataloader:
            data, labels = batch
            data_batch = tokenize(data)

            labels = torch.transpose(labels, 0, 1)

            label_1 = labels[0].to(device)
            label_2 = labels[1].to(device)

            input_ids = data_batch['input_ids'].to(device)
            attention_mask = data_batch['attention_mask'].to(device)

            if c == 'class_1':
                y_true_1.extend(label_1.cpu().numpy())
                label_1.to(device)
                predictions = model_connected(input_ids, attention_mask)
                predictions = predictions[c]
                predictions = torch.argmax(predictions, dim=1)
                y_pred_1.extend(predictions.cpu().numpy())
            else:
                y_true_2.extend(label_2.cpu().numpy())
                label_2.to(device)
                predictions = model_connected(input_ids, attention_mask)
                predictions = predictions[c]
                predictions = torch.argmax(predictions, dim=1)
                y_pred_2.extend(predictions.cpu().numpy())

            if c == 'class_1':
                metric.add_batch(predictions=predictions, references=label_1)
            else:
                metric.add_batch(predictions=predictions, references=label_2)
        if c == 'class_1':
            acc_1 = metric.compute()['accuracy']
        else:
            acc_2 = metric.compute()['accuracy']
    print(f'CLASS_1 accuracy: {acc_1}')
    print(f'CLASS_2 accuracy: {acc_2}')
    return acc_1, acc_2, y_true_1, y_true_2, y_pred_1, y_pred_2