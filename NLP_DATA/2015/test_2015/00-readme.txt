
			       STS 2015

      PILOT SUBTASK on Interpretable Semantic Textual Similarity
			(via Segment Alignment)


                      Test - Evaluation Package


This set of files comprises:

- the test input files (no chunks)

  STSint.input.headlines.sent1.txt
  STSint.input.headlines.sent2.txt
  STSint.input.images.sent1.txt
  STSint.input.images.sent2.txt

- the test input files with gold chunks

  STSint.input.headlines.sent1.chunk.txt
  STSint.input.headlines.sent2.chunk.txt
  STSint.input.images.sent1.chunk.txt
  STSint.input.images.sent2.chunk.txt

- the gold standard files

  STSint.gs.headlines.wa
  STSint.gs.images.wa

- the scripts for evaluation

  evalF1.pl 
  wellformed.pl 

  For example:
  $ perl evalF1.pl gs-file.wa system-file.wa 

- the code for the baseline

  STS15_task2_baseline.tar.gz


See train data release for more details.


Authors
-------

Eneko Agirre
Montse Maritxalar
German Rigau
Larraitz Uria
