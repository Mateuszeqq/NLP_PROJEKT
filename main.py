import argparse

from torch import nn, optim
from torch.utils.data import DataLoader
import torch

from src.models.classifiers import ClassifierSeparate
from src.models.dataset import MyDataset, MyDatasetConnected
from src.operations.train_eval import train_model
from src.processing.data import get_df, collate_fn


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train-path', default='train/processed_headlines.xml',
                    help='override a path for the train dataset')
parser.add_argument('--test-path', default='test/processed_headlines.xml',
                    help='override a path for the test dataset')


parser.add_argument('--batch-size', default=64,
                    help='size of mini batch used during training')
parser.add_argument('--epochs', default=20,
                    help='number of epochs')
parser.add_argument('--learning-rate', default=3e-4,
                    help='learning rate used during training')

args = parser.parse_args()
print(args)

train_path = args.train_path
test_path = args.test_path
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

print(learning_rate)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    df_train, y_train = get_df(train_path, labels=1)
    df_test, y_test = get_df(test_path, labels=1)

    train_dataset = MyDataset(df_train['text_source'].values, df_train['text_translation'].values, y_train.values, 1)
    test_dataset = MyDataset(df_test['text_source'].values, df_test['text_translation'].values, y_test.values, 1)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_separate = ClassifierSeparate(labels=1)
    model_separate = model_separate.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_separate = optim.Adam(model_separate.parameters(), lr=learning_rate)

    train_model(
        model=model_separate,
        optimizer=optimizer_separate,
        epochs=epochs,
        train_dataloader=train_dataloader,
        criterion=criterion,
        device=device,
    )
