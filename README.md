 # Table2Vec: Neural Word and Entity Embeddings for Table Population and Retrieval.
This repository contains resources developed within the master thesis:

 > Table2Vec: Neural Word and Entity Embeddings for Table Population and Retrieval.

The programming language is [Python](https://www.python.org/).  Table embeddings are built using [TensorFlow](https://www.tensorflow.org/tutorials/word2vec).

 ## Table2Vec
 There are hundreds of millions of tables in web pages.  These tables are much richer sources of structured knowledge than free-format text.

Table2Vec is a novel approach that employs neural language modeling to embed different table elements into semantic vector spaces, which can benefit table-related retrieval tasks.

## Dataset


 - The table corpus [Wikipedia Tables](http://websail-fe.cs.northwestern.edu/TabEL/),  which consists of **1.6M** high-quality relational tables in total.  The statistics is as follows:

Core column | Tables in total | Tables* in total |
|-----------| --------------- | ---------------- |
 existing entities | 726,913 | 212,923   |
 60% entities | 556,644 | 139,572  |
 80% entities| 483,665 | 119,166 |
 100% entities | 425,236 | 78,611  |
 100% unique entities | 376,213 | 53,354  |

>  Table* represents the tables that have more than 5 rows and 3 columns. Core column refers to the left most column.

 - The `data/queries.txt` file contains the search queries.

## Table embeddings

> Different table embeddings and their training parameters

Embedding | Total terms | Unique terms | Negative samples | Window size |
-----------------------   | ----------- | ---------- | -------------- | ---------------- |
Table2VecW | 200,157,990 | 1,829,874 | 25 | 5 |
Table2VecH | 7,962,443 | 339,433 | 25 | 20  |
Table2VecE | 24,863,683 | 2,159,467 | 25 | 50 |
Table2VecE* | 5,367,837 | 1,285,708 | 25 | 50 |

 ## Functionality
Table2Vec currently supports three table-related tasks:

 - Table retrieval
 - Row population
 - Column population


## Methods and results:

>The evaluation is undertaken by [trec_eval](https://github.com/usnistgov/trec_eval). For row population, the run files are too big to added on Github.

 1. Row population :

Methods  |    1    |    2    |    3    |   4   |   5   |
-------  | -------- | ------- | -------- | -------- | ------- |
BL1  |  0.4360 | 0.4706 | 0.4788| 0.4786 | 0.4711|
BL2  |  0.2612 | 0.2778 | 0.2845 | 0.2846| 0.2817 |
BL3  |  0.2912 | 0.3024 | 0.3028 | 0.2987| 0.2910 |
Table2VecE* | 0.4982 | 0.5522 | 0.5598 | 0.5543| 0.5476 |
BL1 + Table2VecE*  |**0.5581** | **0.6147** | **0.6400**  | **0.6524** | **0.6533** |
BL2 + Table2VecE* | 0.5461 | 0.6027 | 0.6187 | 0.6217 | 0.6223 |
BL3 + Table2VecE* | 0.5487 | 0.6049 | 0.6218 | 0.6249 | 0.6251 |


 2. Column population  ( the `runfile/CP/` folder contains the run files by various methods ):

Methods  |      Runfile       |    1    |    2    |    3    |
-------  |-----------------  | -------- | ------- | -------- |
Baseline  | `baseline_R1(2,3).txt` | 0.2507 | 0.2845 | 0.2852 |
Baseline + Table2VecH  | `combined_R1(2,3).txt`   | **0.2551** | **0.3322** | **0.4000** |


 3. Table retrieval ( the `runfile/TR/` folder contains the run files by various methods):

| Method |Runfile | NDCG@5 | NDCG@10 | NDCG@15 | NDCG@20 |
|---------------------|-----------------------   | ------ | ------- | ------- | ------- |
|Baseline             |      `gt.txt`            | 0.5527 | 0.5456  | 0.5738  | 0.6031  |
|Baseline + Word2Vec  | `w2v.txt`     | 0.5954 | 0.6006  | 0.6315  | 0.6588  |
Baseline + Graph2Vec|    `g2v.txt`      | 0.5844 | 0.5764  | 0.6128  | 0.6340  |
Baseline + Table2VecW |  `t2vW.txt` | 0.5974 | 0.6096 | 0.6312 | 0.6505 |
Baseline + Table2VecE | `t2vE.txt`  | 0.5602 | 0.5569| 0.5760| 0.6161  |
