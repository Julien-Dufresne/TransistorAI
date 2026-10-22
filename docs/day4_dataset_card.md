**Dataset overview**: English–German translation, technical and news domains.

**Sources**: 
KDE4 from OPUS : version 2 (KDE4.de-en in Moses format), http://opus.nlpl.eu/KDE4-v2.php, licence can be found below.  
newstest2019 from WMT19: version 2019, https://www.statmt.org/wmt19/, licence can be found below.
It is used exclusively for evaluation, not fine tuning. 

**Preprocessing**: Encoding repairs using ftfy, Unicode normalization, filtering and deduplication.

**Dataset statistics**: 
TRAIN
sentence_pairs: 109447
avg_source_words: 10.75
avg_target_words: 10.32
max_source_words: 100
max_target_words: 100
duplicate_pairs: 0

VALIDATION
sentence_pairs: 13615
avg_source_words: 10.8
avg_target_words: 10.35
max_source_words: 100
max_target_words: 100
duplicate_pairs: 0

TECHNICAL_TEST
sentence_pairs: 13652
avg_source_words: 10.65
avg_target_words: 10.28
max_source_words: 100
max_target_words: 99
duplicate_pairs: 0

GENERAL_TEST
sentence_pairs: 1997
avg_source_words: 21.05
avg_target_words: 21.08
max_source_words: 63
max_target_words: 73
duplicate_pairs: 0

**Splitting methodology**: 
Splitting the technical dataset: 80% training, 10% validation, 10% technical testing


**Quality checks**: 
--- TRAIN ---
Sentence pairs: 109,447
Duplicate pairs: 0
Missing values: 0

--- VALIDATION ---
Sentence pairs: 13,615
Duplicate pairs: 0
Missing values: 0

--- TECHNICAL_TEST ---
Sentence pairs: 13,652
Duplicate pairs: 0
Missing values: 0

--- GENERAL_TEST ---
Sentence pairs: 1,997
Duplicate pairs: 0
Missing values: 0

--- CROSS-DATASET OVERLAP ---
train vs validation: 0 shared source sentences
train vs technical_test: 0 shared source sentences
train vs general_test: 0 shared source sentences
validation vs technical_test: 0 shared source sentences
validation vs general_test: 0 shared source sentences
technical_test vs general_test: 0 shared source sentences

**Limitations**: Still need to check:
- More manual checks
- accents
- special characters
- long sentences

*Potential translation errors, short UI strings, domain differences and remaining near-duplicates.*



**Licences**: 
 Corpus Name: KDE4
     Package: KDE4.de-en in Moses format
     Website: http://opus.nlpl.eu/KDE4-v2.php
     Release: v2
Release date: Sat Dec 22 21:08:46 EET 2018

This corpus is part of OPUS - the open collection of parallel corpora
OPUS Website: http://opus.nlpl.eu

Please cite the following article if you use any part of the corpus in your own work: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)

A parallel corpus of KDE4 localization files (v.2).


The data set comes with the same license
as the original sources.
Please, check the information about the source
that is given on
http://opus.nlpl.eu/KDE4-v2.php

Licence for newstest2019 from WMT19:
The data released for the WMT19 news translation task can be freely used for research purposes, we just ask that you cite the WMT19 shared task overview paper, and respect any additional citation requirements on the individual data sets. For other uses of the data, you should consult with original owners of the data sets.

