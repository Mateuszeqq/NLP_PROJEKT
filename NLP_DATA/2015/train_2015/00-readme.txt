
			       STS 2015

      PILOT SUBTASK on Interpretable Semantic Textual Similarity
			(via Segment Alignment)

			    TRAIN DATASET
		      (v03, totalling 753 pairs)

        NEW: evaluation script has been updated, see below
				   

This set of files describes the train DATASET for the first PILOT
SUBTASK on Interpretable Semantic Textual Similarity via Segment Alignment.

The train dataset contains the following:

  00-README.txt 		         this file

Two datasets, headlines and images

  STSint.input.headlines.sent1.txt       First sentence in input sentence pairs (headlines)
  STSint.input.headlines.sent2.txt       Second sentence in input sentence pairs (headlines)
  STSint.input.images.sent1.txt          First sentence in input sentence pairs (images)
  STSint.input.images.sent2.txt          Second sentence in input sentence pairs (images)

  STSint.input.headlines.sent1.chunk.txt First sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.input.headlines.sent2.chunk.txt Second sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.input.images.sent1.chunk.txt    First sentence in input sentence pairs, with gold standard chunks (images)
  STSint.input.images.sent2.chunk.txt    Second sentence in input sentence pairs, with gold standard chunks (images)

  STSint.gs.headlines.wa		 Gold standard alignment for each sentence pair in input (headlines)
  STSint.gs.images.wa			 Gold standard alignment for each sentence pair in input (images)

Scripts

  wellformed.pl                          Script to check for well-formed output
  evalF1.pl                              Official evaluation script

Sample toy files in evalsamples directory to check evaluation script

  evalsamples/gs.wa			 Sample toy gold standard file (one repeated pair)
  evalsamples/output.gschunk.wa		 Sample toy system output, using gold-standard chunks
  evalsamples/output.syschunk.wa	 Sample toy system output, using sys chunks


Introduction
------------

The train data was sampled from STS 2014 headline and image sentence
pairs.

Each dataset was sampled from previous STS datasets:

- headlines: Headlines mined from several news sources by European
  Media Monitor using their RSS feed.
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html

- images: The Image Descriptions data set is a subset of the Flickr
  dataset presented in (Rashtchian et al., 2010), which consisted on
  8108 hand-selected images from Flickr, depicting actions and events
  of people or animals, with five captions per image. The image
  captions of the data set are released under a CreativeCommons
  Attribution-ShareAlike license.



Input format
------------

The input consists of two files:

- a file with the first sentences in each pair
- a file with the second sentences in each pair

The sentences are tokenized.

Please check STSint.input.*.sent[12].txt

Participants can also use the input sentences with gold standard chunks:

- a file with the first sentences in each pair, with [ and ] to mark chunks
- a file with the second sentences in each pair, with [ and ] to mark chunks

Please check STSint.input.*.sent[12].chunk.txt



Gold Standard alignment
-----------------------

The gold standard annotation format is the word alignment format (.wa
files), an XML file as produced by
https://www.ldc.upenn.edu/language-resources/tools/ldc-word-aligner

We slightly modified the format to also include the score. Each
alignment is reported in one line as follows:

  token-id-seq1 <==> token-id-seq2 // type // score // comment

where:

  token-id-seq1 is a sequence of token indices (starting at 1) for the
     chunk in sentence 1 (or 0 if the chunk in sentence 2 is not
     aligned or is ALIC) 

  token-id-seq2 is a sequence of token indices (starting at 1) for the
     chunk in sentence 2 (or 0 if the chunk in sentence 1 is not
     aligned or is ALIC) 

  type is composed of one of the obligatory labels, concatenated to
     the optional ones by '_'

  score is a number from 0 to 5, or NIL (if type label is NOALI or ALIC) 

  comment is any string

Please check STSint.gs.*.wa 



Answer format
--------------

The same format as the gold standard alignment has to be used.  Only
the alignment section of the XML file will be used. The source and
target sections will be ignored (so any system using different token
numbers would be penalized). The sentence id is very important, as it
will be used to find the corresponding gs pair.

Please check STSint.output.wa

You can check for well-formedness using the provided script as follows:

    $ ./wellformed.pl STSint.gs.headlines.wa
    $ ./wellformed.pl STSint.gs.images.wa
    $ ./wellformed.pl evalsamples/gs.wa
    $ ./wellformed.pl evalsamples/output.gschunk.wa
    $ ./wellformed.pl evalsamples/output.syschunk.wa

Answer files which fail for well-formedness using the script above
will be automatically discarded from evaluation.

The same program prints several statistics:

    $ ./wellformed.pl STSint.gs.headlines.wa --stats=1
    $ ./wellformed.pl STSint.gs.images.wa --stats=1


Scoring
-------

The official evaluation is based on (Melamed, 1998), which uses the F1
of precision and recall of token alignments (in the context of
alignment for Machine Translation). Fraser and Marcu (Fraser and
Marcu, 2007) argue that F1 is a better measure than Alignment Error
Rate.

The idea is that, for each pair of chunks that are aligned, we
consider that any pairs of tokens in the chunks are also aligned with
some weight. The weight of each token-token alignment is the inverse
of the number of alignments of each token (so-called fan out factor,
Melamed, 1998). Precision is measured as the ratio of token-token
alignments that exist in both system and gold standard files, divided
by the number of alignments in the system. Recall is measured
similarly, as the ratio of token-token alignments that exist in both
system and gold-standard, divided by the number of alignments in the
gold standard. Precision and recall are evaluated for all alignments
of all pairs in one go.

The script provides four evaluation measures:

- F1 where alignment type and score are ignored
- F1 where alignment types need to match, but scores are
  ignored. Match is quantified using Jaccard, as there can be multiple
  tags (FACT,POL).
- F1 where alignment type is ignored, but each alignment is penalized
  when scores do not match
- F1 where alignment types need to match, and each alignment is
  penalized when scores do not match. Match is quantified using
  Jaccard, as there can be multiple tags (FACT,POL). In addition the
  following special cases are catered for:
  . there is no type penalty between tags {SPE1, SPE2, REL, SIMI} when
    scores are (0-2] 
  . there is no type penalty between EQUI and SIMI/SPE with score 4
    (F1AST) 

When run with the debugging flag on, the script prints detailed
scores. It also computes the precision and recall scores by pair (for
illustration purposes only).

See the header of evalF1.pl for the exact formula and change log.

Examples of use:

   # check a system which is based on correct chunks
   $ ./evalF1.pl evalsamples/gs.wa evalsamples/output.gschunk.wa
   $ ./evalF1.pl examples/gs.wa evalsamples/gs.wa evalsamples/output.syschunk.wa

   # detailed scores, including illustrative performance per pair 
   $ ./evalF1.pl --debug=1 evalsamples/gs.wa evalsamples/output.gschunk.wa
   $ ./evalF1.pl --debug=1 evalsamples/gs.wa evalsamples/output.syschunk.wa


Authors
-------

Eneko Agirre
Montse Maritxalar
German Rigau
Larraitz Uria


References
----------

Dan Melamed. 1998. Manual annotation of translational equivalence: The
   blinker project. Technical Report 98-07, Institute for Research in
   Cognitive Science, Philadelphia

Alexander Fraser and Daniel Marcu. Measuring Word Alignment Quality
  for Statistical Machine Translation. Computational Linguistics 2007
  33:3, 293-303.

Rashtchian, C., Young, P., Hodosh, M., and Hockenmaier, J.
  Collecting Image Annotations Using Amazon's Mechanical Turk.  In
  Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and
  Language Data with Amazon's Mechanical Turk. 2010.
