Unlike the original word2vec program which is self-contained,
the word2vecf program assumes some precomputations.

In particular, word2vecf DOES NOT handle vocabulary construction, and does
not read an unprocessed input.

The expected files are:
word_vocabulary:
   file mapping integers to strings (and has a count for each word)
   not used by the program, but needed for interpretting its output.
context_vocabulary:
   file mapping integers to string, and has a count for each feature.
   used for constructing the sampling table for the negative training.
training_data:
   binary file with words and their contexts.
   the format is pairs of integers (32bit each), first integer is the word,
   second integer is the context.
   if we want to prefer some contexts over the others, we should construct the
   training data to contain the bias.
   the data at even indices are the words, the data at odd indices are the
   contexts (zero based).

also needed: word vocabulary size, contexts vocabulary size.


Tools for computing distributed representtion of words
------------------------------------------------------

We provide an implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous
Bag-of-Words or the Skip-Gram neural network architectures. The user should to specify the following:
 - desired vector dimensionality
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model
 - training algorithm: hierarchical softmax and / or negative sampling
 - threshold for downsampling the frequent words 
 - number of threads to use
 - the format of the output word vector file (text or binary)

Usually, the other hyper-parameters such as the learning rate do not need to be tuned for different training sets. 

The script demo-word.sh downloads a small (100MB) text corpus from the web, and trains a small word vector model. After the training
is finished, the user can interactively explore the similarity of the words.

More information about the scripts is provided at https://code.google.com/p/word2vec/

