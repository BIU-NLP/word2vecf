//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


const int vocab_hash_size = 50000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct vocabulary {
   struct vocab_word *vocab;
   int *vocab_hash;
   long long vocab_max_size; //1000
   long vocab_size;
};

char train_file[MAX_STRING], output_file[MAX_STRING], cvocab_file[MAX_STRING], wvocab_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, use_position = 0;
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0, dumpcv = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

//{{{ Hash
#include "stdint.h" /* Replace with <stdint.h> if appropriate */
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif


// hash from: http://www.azillionmonkeys.com/qed/hash.html
uint32_t FastHash(const char * data, int len) {
uint32_t hash = len, tmp;
int rem;

    if (len <= 0 || data == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
} //}}}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) break;
      else continue; 
      //if (ch == '\n') {
      //  strcpy(word, (char *)"</s>");
      //  return;
      //} else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
inline int GetWordHash(struct vocabulary *v, char *word) {
  unsigned long long hash = 0;
  char *b = word;
  //for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  //hash = FastHash(word, strlen(word)) % vocab_hash_size;
  while (*b != 0) hash = hash * 257 + *(b++);
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(struct vocabulary *v, char *word) {
  unsigned int hash = GetWordHash(v, word);
  while (1) {
    if ((v->vocab_hash)[hash] == -1) return -1;
    if (!strcmp(word, v->vocab[v->vocab_hash[hash]].word)) return v->vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(struct vocabulary *v, FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(v, word);
}

// Adds a word to the vocabulary
int AddWordToVocab(struct vocabulary *v, char *word) {
  //static long collide = 0;
  //static long nocollide = 0;
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  v->vocab[v->vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(v->vocab[v->vocab_size].word, word);
  v->vocab[v->vocab_size].cn = 0;
  v->vocab_size++;
  // Reallocate memory if needed
  if (v->vocab_size + 2 >= v->vocab_max_size) {
    v->vocab_max_size += 1000;
    v->vocab = (struct vocab_word *)realloc(v->vocab, v->vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(v, word);
  //if (v->vocab_hash[hash] != -1) { collide += 1; } else { nocollide += 1; }
  //if ((collide + nocollide) % 100000 == 0) printf("%d %d %f collisions\n\n",collide, nocollide, (float)collide/(collide+nocollide));
  while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  v->vocab_hash[hash] = v->vocab_size - 1;
  return v->vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortAndReduceVocab(struct vocabulary *v, int min_count) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&(v->vocab[1]), v->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
  size = v->vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (v->vocab[a].cn < min_count) {
      v->vocab_size--;
      free(v->vocab[v->vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(v, v->vocab[a].word);
      while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      v->vocab_hash[hash] = a;
      train_words += v->vocab[a].cn;
    }
  }
  v->vocab = (struct vocab_word *)realloc(v->vocab, (v->vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < v->vocab_size; a++) {
    v->vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    v->vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct vocabulary *v) {
   printf("reducevocab\n");
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < v->vocab_size; a++) if (v->vocab[a].cn > min_reduce) {
    v->vocab[b].cn = v->vocab[a].cn;
    v->vocab[b].word = v->vocab[a].word;
    b++;
  } else free(v->vocab[a].word);
  v->vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
  for (a = 0; a < v->vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(v, v->vocab[a].word);
    while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    v->vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

struct vocabulary *CreateVocabulary() {
   struct vocabulary *v = malloc(sizeof(struct vocabulary));
   long long a;
   v->vocab_max_size = 1000;
   v->vocab_size = 0;

   v->vocab = (struct vocab_word *)calloc(v->vocab_max_size, sizeof(struct vocab_word));

   v->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
   for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
   return v;
}

void SaveVocab(struct vocabulary *v, char *save_vocab_file) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < v->vocab_size; i++) fprintf(fo, "%s %lld\n", v->vocab[i].word, v->vocab[i].cn);
  fclose(fo);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  char context[MAX_STRING];
  FILE *fin;// = stdin;
  long long a, i;
  int wi, ci;
  struct vocabulary *wv = CreateVocabulary();
  struct vocabulary *cv = CreateVocabulary();
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  AddWordToVocab(wv, (char *)"</s>");
  AddWordToVocab(cv, (char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    ReadWord(context, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(wv,word);
    if (i == -1) {
      a = AddWordToVocab(wv,word);
      wv->vocab[a].cn = 1;
    } else wv->vocab[i].cn++;

    i = SearchVocab(cv,context);
    if (i == -1) {
      a = AddWordToVocab(cv,context);
      cv->vocab[a].cn = 1;
    } else cv->vocab[i].cn++;

    if (wv->vocab_size > vocab_hash_size * 0.7) ReduceVocab(wv);
    if (cv->vocab_size > vocab_hash_size * 0.7) ReduceVocab(cv);
  }
  SortAndReduceVocab(wv,min_count);
  SortAndReduceVocab(cv,min_count);
  if (debug_mode > 0) {
    printf("WVocab size: %lld\n", wv->vocab_size);
    printf("CVocab size: %lld\n", cv->vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
  SaveVocab(wv, wvocab_file);
  SaveVocab(cv, cvocab_file);
  
  /////////////////////////////
  printf("\nSaved reduced vocabs, writing binary output\n\n");
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  FILE* fout = fopen(output_file, "wb");
  if (fout == NULL) {
    printf("ERROR: outputfile cannot be created!\n");
    exit(1);
  }
  train_words = 0;
  while (!feof(fin)) {
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    wi = ReadWordIndex(wv, fin);
    ci = ReadWordIndex(cv, fin);
    if (ci < 0 || wi < 0) continue;
    fwrite(&wi,sizeof(int),1,fout);
    fwrite(&ci,sizeof(int),1,fout);
  }
  fclose(fin);
  fclose(fout);

  //TODO clean vocabularies
}


/*
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}*/

void TrainModel() {
   printf("sizeof int:%d\n", sizeof(int));
  printf("Starting training using file %s\n", train_file);
  LearnVocabFromTrainFile();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
    printf("\t-dumpcv 1\n");
    printf("\t\tDump the context vectors, in file <output>.context\n");
    printf("\t-pos 1\n");
    printf("\t\tInclude sequence position information in context.\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  output_file[0] = 0;
  wvocab_file[0] = 0;
  cvocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-dumpcv", argc, argv)) > 0) dumpcv = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-pos", argc, argv)) > 0) use_position = 1;
  if ((i = ArgPos((char *)"-wvocab", argc, argv)) > 0) strcpy(wvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cvocab", argc, argv)) > 0) strcpy(cvocab_file, argv[i + 1]);
  if (output_file[0] == 0 || cvocab_file[0] == 0 || wvocab_file[0] == 0) {
     printf("-output or -cvocab or -wvocab argument is missing\n\n");
     return 0;
  };
  if (dumpcv && negative == 0) {
     printf("-dumpcv requires negative training.\n\n");
     return 0;
  };
  if (dumpcv && (use_position > 0)) {
     printf("-dumpcv cannot run with use_position yet.\n\n");
     return 0;
  };
  if ((hs > 0 || cbow > 0) && (use_position > 0)) {
     printf("-use_position require skip-gram negative-sampling training.\n\n");
     return 0;
  };
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
