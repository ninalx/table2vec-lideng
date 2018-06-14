 # Table2Vec: Neural Word and Entity Embeddings for Table Population and Retrieval.  
This repository contains resources developed within the master thesis:  
  
 > Table2Vec: Neural Word and Entity Embeddings for Table Population and Retrieval. 

The programming language is [Python](https://www.python.org/).  Table embeddings are built using [TensorFlow](https://www.tensorflow.org/tutorials/word2vec).
 
 ## Table2Vec
 There are hundreds of millions of tables in web pages.  These tables are much richer sources of structured knowledge than free-format text. 
 
Table2Vec is a novel approach that employs neural language modeling to embed different table elements into semantic vector spaces, which can benefit table-related retrieval tasks. 

## Dataset  
  
The table corpus [Wikipedia Tables](http://websail-fe.cs.northwestern.edu/TabEL/),  which consists of **1.6M** high-quality relational tables in total. 



 ## Functionality
Table2Vec currently supports three table-related tasks:

 - Table retrieval 
 - Row population
 - column population

### Vector Representation of Words or Entities:  
  
> This is the done part and we use skip-gram model to train different embeddings.  
  
vocabulary statistics:  
  
Embeddings Type          | Total Words     | Unique Words    |   
-----------------------   | -------------   | -------------   |  
word embedding(all)      | 200,157,990     | 1,829,874       |   
entity embedding(new)     | 24,746,422      | 2,160,311       |   
heading embedding (words) | 12,392,732      | 133,803         |   
heading embedding (cells) | 7,962,443       | 339,433         |  
  
embedding parameters:  
  
Embeddings Type          | Window Size | Batch Size | Embedding Size | Negative Samples |  
-----------------------   | ----------- | ---------- | -------------- | ---------------- |  
word embedding(all)      | 5           | 500        | 200            | 25               |  
~~entity embedding~~      | ~~50~~      | ~~500~~    | ~~200~~        | ~~25~~           |  
entity embedding(1st)     | 50          | 500        | 200            | 25               |  
heading embedding (words) | 10          | 500        | 200            | 25               |  
heading embedding (cells) | 5           | 500        | 200            | 25               |  
  
### Evaluation  
  
> This is an ongoing part.  
  
Evaluation is undertaken with a program called trec_eval:  
  
Embeddings Type          | NDCG@5 | NDCG@10 | NDCG@15 | NDCG@20 |  
-----------------------   | ------ | ------- | ------- | ------- |  
baseline                  | 0.5593 | 0.5498  | 0.5759  | 0.6039  |  
word embedding(all)      | 0.5974 | 0.6096  | 0.6312  | 0.6505  |  
entity embedding          | ###### | ######  | ######  | ######  |  
heading embedding (words) | ###### | ######  | ######  | ######  |  
heading embedding (cells) | ###### | ######  | ######  | ######  |  
  
row population: MAP/MRR  
  
Methods               |        1        |       2        |       3        |       4       |        5      |  
--------------------  | --------------- | -------------- | -------------- | ------------- | ------------- |  
(1)entity embeddings  | 0.4982/0.7623   | 0.5178/0.8081  | ### | ### | ### |  
(2)relations, 0.5     | 0.4963/0.6857   | 0.5469/0.7297  | 0.5687/0.7415  | 0.5734/0.7294 | 0.5693/0.7274 |  
(3)wlm, 0.5           | 0.4674/0.6246   | 0.5154/0.6901  | 0.5293/0.6930  | 0.5331/0.6861 | 0.5258/0.6789 |  
(4)jaccard, 0.5       | 0.4905/0.6731   | 0.5427/0.7086  | 0.5617/0.7270  | 0.5662/0.7098 | 0.5609/0.7058 |  
(1)&(3), 0.6          | 0.5159/0.7625   | 0.5248/0.7920  |###|###|###|  
(1)&(4), 0.6          | 0.4739/0.7000   |  ### |###|###|###|   
smarttable assistance*| **0.5922**/0.7727   | **0.6260**/0.8000  | 0.6339/0.7849  | 0.6348/0.7800 | 0.6310/0.7630 |  
(1)&(2),this paper    | 0.5576/0.7414   | 0.6143/0.8141  | **0.6398**/0.8424  | **0.6517**/**0.8427** | **0.6531**/0.8372 |