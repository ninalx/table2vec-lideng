#
# source code for training table embeddings
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import math
import json
import numpy as np
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type = 'BFC'
# session = tf.Session(config=config)

#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", "Path", "Directory to write the model.")
flags.DEFINE_string(
    "train_data", "Path",
    "Training data.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 50,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
# window_size = 5
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")

## newly added -- random validation set to sample nearest neighbors, valid_size, valid_window
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.

flags.DEFINE_integer("valid_size", 16,
                     "Random set of words to evaluate similarity on.")
flags.DEFINE_integer("valid_window", 100,
                     "Only pick dev samples in the head of the distribution.")


flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # evaluation sample by predicting nearby words
    self.valid_size = FLAGS.valid_size
    self.valid_window = FLAGS.valid_window

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    #self.save_vocab()
    #self.save_emb()

  def build_graph(self):
    """Build the model graph."""
    opts = self._options
    valid_examples = np.random.choice(opts.valid_window, opts.valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # The training data. A text file.
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels,) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                    batch_size=opts.batch_size,
                                                    window_size=opts.window_size,
                                                    min_count=opts.min_count,
                                                    subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)

    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i

    # Declare all variables we need.

    # Input words embedding: [vocab_size, emb_dim]: this is the embedding we care about.
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    print("w-in = embedding", w_in)
    print("w-out = softmax weight", w_out)
    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train_word2vec(w_in,
                                          w_out,
                                          examples,
                                          labels,
                                          lr,
                                          vocab_count=opts.vocab_counts.tolist(),
                                          num_negative_samples=opts.num_samples)


    self._w_in = w_in
    self._w_out = w_out
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.global_step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed
    self._valid_dataset = valid_dataset
    self._valid_examples = valid_examples


  def save_emb(self, index):
    """save embeddings, with (key, value) -> (word, embeddings)."""
    opts = self._options

    nemb = tf.nn.l2_normalize(self._w_in, 1)
    nemb = self._session.run(nemb)
    dictionary = dict()
    with open(os.path.join(opts.save_path,"/home/stud/lideng/entity_test_gt/result_200/entity_emb_%i.json") %index, "w") as f:
      for i in range(opts.vocab_size):
        vocab_word = tf.compat.as_text(opts.vocab_words[i])
        dictionary[vocab_word] = nemb[i].tolist()
      json.dump(dictionary, f, indent=4, separators=(',', ': '))


  def nearby_words(self):
    opts = self._options
    norm = tf.sqrt(tf.reduce_sum(tf.square(self._w_in), 1, keep_dims=True))
    #print(norm)
    normalized_embeddings = self._w_in / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, self._valid_dataset)
    similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
    sim = similarity.eval()
    for i in range(opts.valid_size):
        valid_word = self._id2word[self._valid_examples[i]]
        #print('type valid_word',type(valid_word))
        valid_word = valid_word.decode('utf-8',errors ='ignore')
        # print("valid_word",valid_word)
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        #print(nearest)
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = self._id2word[nearest[k]]
            close_word = close_word.decode('utf-8',errors='ignore')
            log_str = '%s %s,' % (log_str,close_word)
        log_str = log_str.encode('utf-8')
        print(log_str)


  def loss_eval(self):
    # Construct the variables for the NCE loss
    opts = self._options
    nce_weights = tf.Variable(
        tf.truncated_normal([opts.vocab_size, opts.emb_dim],
                            stddev=1.0 / math.sqrt(opts.emb_dim)))
    nce_biases = tf.Variable(tf.zeros([opts.vocab_size]))

    # define loss
    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=self._labels,
                     inputs=self._w_out,
                     num_sampled=opts.batch_size,
                     num_classes=opts.vocab_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)
    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    self.saver = tf.train.Saver()

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in range(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(5)  # Reports our progress once a while.
      (epoch, step, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()


  def nearby(self, filename, index, num=101):
    """Prints out nearby words given a list of words."""
    seed_entity = json.load(open(filename))
    seed_dict = dict()
    for table_id in seed_entity:
      words = seed_entity[table_id]
      dic1 = dict()
      ids = np.array([self._word2id.get(x.encode('utf-8'), 0) for x in words])
      vals, idx = self._session.run(
                        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
      for i in range(len(words)):
        neary_words = []
        for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
          if words[i].encode('utf-8') == self._id2word[neighbor]:
            continue
          else:
            neary_words.append(self._id2word[neighbor].decode('utf-8'))
        dic1[words[i]] = neary_words
      seed_dict[table_id] = dic1
    with open("/home/stud/lideng/entity_test_gt/result_200/seed_rel_entity_%i.json"%index, 'w') as f:
      json.dump(seed_dict, f, indent=4, separators=(',', ': '))


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

def main(_):
  """Train a embedding model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
      sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as session:

      with tf.device("/device:GPU:0"):
          model = Word2Vec(opts, session)
          print("================================")
      index = 0
      for _ in range(opts.epochs_to_train):
          index += 1
          #print("epoch", index, "- training the model...")
          model.train()  # Process one epoch
          print("================================")
          if index % 5 == 0:
              print("==saving the models: lideng.py==")
              model.nearby("/home/stud/lideng/entity_test_gt/seed_entity_order.json",index)
              model.save_emb(index)

      if FLAGS.interactive:
          _start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
