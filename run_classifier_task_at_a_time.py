
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_task_at_a_time
import optimization
import tokenization
import tensorflow as tf
import preprocess_corpus
import numpy as np
from tensorflow.python.framework import ops
from skopt import gp_minimize

def diff_loss(A, B):
    return tf.norm(tf.matmul(tf.transpose(tf.to_float(A)), tf.to_float(B)))**2 #this wont work for tensors rank 1
#- WTF WITH THAT


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()


names = ['Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
         'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen',
            'Kindle_Store', 'Sports_and_Outdoors', 'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
             'Toys_and_Games',
             'Video_Games', 'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android', 'Office_Products']

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer(
    "common_enc_size", 768,
    "Common encoder units.")
flags.DEFINE_integer(
    "private_enc_size", 768,
    "Private encoder units.")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "adversarial", True,
    "Wherther to use encoders and adversarial module")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("orth_loss_weight", 0.1,
                   "Orth loss weight")

flags.DEFINE_float("discr_loss_weight", 0.05,
                   "Discr loss weight")

flags.DEFINE_float(
    "warmup_proportion", 0.05, #0.1
    "Proportion of training to 12perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 100, #TODO THIS IS WHERE EVAL FREQUENCY IS SET
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, task_id = None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.task_id = task_id


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               task_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.task_id = task_id
    self.is_real_example = is_real_example




def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        task_id = None,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  '''if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))'''
  task_id = example.guid
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      task_id = task_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    features["task_id"] = create_int_feature([int(feature.task_id)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "task_id": tf.FixedLenFeature([], tf.int64)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      #d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d # gotta check if batch only has tasks of one kind! That's what task_id is for

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

task_step = 0

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, task_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling_task_at_a_time.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      task_ids = task_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  current_task_id, output_layer = model.get_pooled_output()
  if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  #assert current_task_id == task_ids[0]
  #current_task_id = 0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  hidden_size = output_layer.shape[-1].value
  weights, biases = [], []
  if FLAGS.adversarial:
      private_enс_weights, private_enc_biases = [], []

  for i in range(len(names)):
      with tf.variable_scope(str(i) + "_th_task"):

          if FLAGS.adversarial:
              weights.append(tf.get_variable(
              "output_weights", [num_labels, FLAGS.common_enc_size+FLAGS.private_enc_size], #2*hidden_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02)))
          else:
              weights.append(tf.get_variable(
                  "output_weights", [num_labels, hidden_size],
                  initializer=tf.truncated_normal_initializer(stddev=0.02)))

          if FLAGS.adversarial:
              private_enс_weights.append((tf.get_variable(
              "priv_enc_weights", [FLAGS.private_enc_size, hidden_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02))))

          biases.append(tf.get_variable(
              "output_bias", [num_labels], initializer=tf.zeros_initializer()))

          if FLAGS.adversarial:
              private_enc_biases.append(tf.get_variable(
              "priv_enc_bias", [FLAGS.private_enc_size], initializer=tf.zeros_initializer()))


  if FLAGS.adversarial:
      with tf.variable_scope("common_encoder"): # ONLY ONE FOR ALL TASKS
        common_encoder = tf.layers.dense(inputs=output_layer,
                                       units=FLAGS.common_enc_size,#hidden_size,
                                       activation=tf.nn.tanh,
                                       use_bias=True,
                                       )
  if FLAGS.adversarial:
      with tf.variable_scope("task_discriminator"):# ONLY ONE FOR ALL TASKS
        discriminator = tf.layers.dense(inputs=flip_gradient(common_encoder),
                                       units=len(names),
                                       #activation=tf.nn.tanh,
                                       use_bias=True,
                                       )

  all_tasks_weights_cube = tf.stack(weights)
  all_tasks_biases_cube = tf.stack(biases)
  if FLAGS.adversarial:
      all_tasks_enc_weights_cube = tf.stack(private_enс_weights)
      all_tasks_enc_biases_cube = tf.stack(private_enc_biases)

  with tf.variable_scope("loss"):

      if FLAGS.adversarial:
          private_encoder = tf.matmul(output_layer, tf.gather(all_tasks_enc_weights_cube, current_task_id, axis=0), transpose_b=True)
          private_encoder = tf.nn.bias_add(private_encoder, tf.gather(all_tasks_enc_biases_cube, current_task_id, axis=0))
          private_encoder = tf.nn.tanh(private_encoder)

          logits = tf.matmul(tf.concat([common_encoder, private_encoder], axis=-1), tf.gather(all_tasks_weights_cube, current_task_id, axis=0), transpose_b=True)
      else:
          logits = tf.matmul(output_layer,
                             tf.gather(all_tasks_weights_cube, current_task_id, axis=0), transpose_b=True)
      logits = tf.nn.bias_add(logits, tf.gather(all_tasks_biases_cube, current_task_id, axis=0))
      probabilities = tf.nn.softmax(logits, axis=-1)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)

      if FLAGS.adversarial:
          discr_one_hot_labels = tf.one_hot(tf.squeeze(task_ids), depth=len(names), dtype=tf.float32)
          discr_log_probs = tf.nn.log_softmax(discriminator, axis=-1)
          discr_loss = tf.reduce_mean(-tf.reduce_sum(discr_one_hot_labels * discr_log_probs, axis = -1))

          orth_loss = diff_loss(common_encoder, private_encoder)
      else:
          discr_loss, orth_loss = 0, 0

      return (current_task_id, loss, per_example_loss, logits, probabilities, discr_loss, orth_loss)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    task_ids = features["task_id"] # seems like input_fn yields batch features and this gives us batch task_ids
    if mode == tf.estimator.ModeKeys.TRAIN:
        assert task_ids.shape.as_list()[0] == FLAGS.train_batch_size
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (current_task_id, task_loss, per_example_loss, logits, probabilities, discr_loss, orth_loss) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, task_ids, label_ids,
        num_labels, use_one_hot_embeddings) # everything here only applies to one of the tasks

    total_loss = task_loss + FLAGS.discr_loss_weight * discr_loss + FLAGS.orth_loss_weight * orth_loss #HYPERPARAMS HERE!!!!!!!!!!!!!!!!

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_task_at_a_time.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    '''tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"  
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)'''

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss, # NOT SURE. WHAT IS THIS FOR? WE RETURN eval_loss, which is task loss
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn




def _my_create_examples(texts_list, labels_list, num_tasks, corpus_length, batch_size):
    examples = []
    i = 0

    rest = corpus_length % batch_size
    corpus_length -= rest # full batches only

    while i <= corpus_length-batch_size: #iterate through dataset
        #texts, labels = [], []
        perm = np.random.permutation(num_tasks)
        #for j in range(num_tasks): #iterate through each task - REPLACE WITH RANDOM TASK CHOICE
        for j in perm:
            for k in range(batch_size):
                text = tokenization.convert_to_unicode(texts_list[j][i+k]).strip( '\n' )
                label = tokenization.convert_to_unicode(labels_list[j][i+k]).strip( '\n' )

                examples.append(
                    InputExample(guid=j, text_a=text, label=label)) # assert must fall j
        i += batch_size

    #print(len(examples), batch_size)
    #print(corpus_length)
    assert len(examples) == corpus_length*num_tasks
    return examples

def hyperparams_wrapper(params):
    #params: batch_size, lr, orth loss weight, discr loss weight

    tf.reset_default_graph()

    FLAGS.adversarial = False

    def convert_params_to_string(params):
        st = "_"
        for param in params:
            st += str(param)[:15]
            st += "_"
        return st

    FLAGS.task_name = "MRPC"
    FLAGS.do_train = True
    FLAGS.do_eval = True
    FLAGS.data_dir = "./glue_data/MRPC"
    FLAGS.vocab_file = "./uncased_L-12_H-768_A-12/vocab.txt"
    FLAGS.bert_config_file = "./uncased_L-12_H-768_A-12/bert_config.json"
    FLAGS.init_checkpoint = "./uncased_L-12_H-768_A-12/bert_model.ckpt"

    FLAGS.max_seq_length = 140 #256
    FLAGS.train_batch_size = params[0]
    FLAGS.learning_rate = params[1]#2e-5
    #FLAGS.orth_loss_weight = params[2]
    #FLAGS.discr_loss_weight = params[3]
    #FLAGS.common_enc_size = params[4]
    #FLAGS.private_enc_size = params[5]
    FLAGS.num_train_epochs = 3.0
    #FLAGS.output_dir = "./outputs/task_at_a_time/hyperparams_search"+convert_params_to_string(params)
    FLAGS.output_dir = "./outputs/task_at_a_time_NO_ADV/hyperparams_search" + convert_params_to_string(params)

    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.logging.set_verbosity(tf.logging.DEBUG)

    '''processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
    }'''

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling_task_at_a_time.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    # if task_name not in processors: #MRPC goes here
    #  raise ValueError("Task not found: %s" % (task_name))

    # processor = processors[task_name]()

    # label_list = processor.get_labels()
    label_list = ["0", "1"]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)  # what is vocab_file??? --vocab_file=$BERT_BASE_DIR/vocab.txt - COMES FROM PRETRAINED BERT

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    gpu_options = tf.GPUOptions(allow_growth=True) #gpu_fraction = 0.3
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True) #TODO implement this!!!!
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=1,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config = tf.ConfigProto(gpu_options=gpu_options),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    train_size = 4000
    test_size = 400

    texts_train_list, texts_test_list, labels_train_list, labels_test_list = [], [], [], []

    if FLAGS.do_train:  # TRUE
        for name in names:
            # train_examples = processor.get_train_examples(FLAGS.data_dir) # INSERT YOUR CODE HERE - PASS LIST OF whatever!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            my_texts_train, my_texts_test, my_labels_train, my_labels_test = preprocess_corpus.preprocess(name,
                                                                                                          "./output",
                                                                                                          remake=False)

            assert len(my_texts_train) == len(my_labels_train) == train_size
            assert len(my_texts_test) == len(my_labels_test) == test_size
            # train_examples_list.append(_my_create_examples(texts=my_texts_train, labels=my_labels_train))
            # test_examples_list.append(_my_create_examples(texts=my_texts_test, labels=my_labels_test))
            texts_train_list.append(my_texts_train)
            texts_test_list.append(my_texts_test)
            labels_train_list.append(my_labels_train)
            labels_test_list.append(my_labels_test)


        train_examples = _my_create_examples(texts_train_list, labels_train_list, len(names), corpus_length=train_size,
                                             batch_size=FLAGS.train_batch_size)  # list of examples
        eval_examples = _my_create_examples(texts_test_list, labels_test_list, len(names), corpus_length=test_size,
                                            batch_size=FLAGS.eval_batch_size)  # list of examples

        train_size = len(train_examples)
        test_size = len(eval_examples)
        num_train_steps = int(
            train_size / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(  # doesnt get examples yet
        bert_config=bert_config,
        num_labels=2,  # labels_list = [0,1]
        init_checkpoint=FLAGS.init_checkpoint,  # pretrained checkpoint
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        # num_tasks = len(names),
        # seq_length = FLAGS.max_seq_length,
    )  # OKAY seems like embs dont need to be one-hot

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,  # False
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # train_file_list = []
    if FLAGS.do_train:
        # for i, name in enumerate(names):
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer,
            train_file)  # put binary-serialized features to the train_file
        #tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(  # THIS FUNCTION RETURNS TRAIN EXAMPLES AND LABELS
            input_file=train_file,  # HERE IS TRAIN FILE
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            # num_tasks = len(names)
        )
        # okay SO HERE GOES THE TRAIN
        # let's say minimum steps is 1000 and after that checkpoints will be saved for each 250 steps. If no decrease for next 4 checks then stop.

        stop_hook = tf.contrib.estimator.stop_if_no_decrease_hook(estimator, "eval_loss", max_steps_without_decrease=400, min_steps = 750, run_every_secs=60)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[stop_hook])

        #estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        #tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            # num_tasks= len(names)

        )

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=120, throttle_secs=0)
        tf.logging.info("start experiment...")

        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )


        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            #print("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                #print(key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    #return (result['eval_loss'], result['eval_accuracy'])
    return result['eval_loss']


def main(_):

  #hp_tuning_result = gp_minimize(hyperparams_wrapper, [[8, 12, 14], (5e-6, 1e-4), (1e-2, 0.5), (1e-3, 1e-1), [276, 424, 576, 768], [276, 424, 576, 768]],
  #                               n_calls=40, x0=[14, 5.48386e-5, 0.5, 0.001, 424, 424],
  #                               verbose=True)
  hp_tuning_result = gp_minimize(hyperparams_wrapper, [[8, 12, 14], (5e-6, 1e-4)],
                                 n_calls=40, x0=[12, 4.5954111903935344e-05],
                                 verbose=True)

  #4 tasks opt is PARAMS [8, 0.00012212957106648347, 0.051983923229111814, 0.0483198888684224] LOSS 0.30725384
  #16 tasks opt is 12_0.0001_0.5_0.1_576_768
  #latest corrupted session reaches opt at hyperparams_search_14_5.48386_0.5_0.001_424_424_, loss is 0.19018593430519104
  #TODO latest adv session reaches opt at hyperparams_search_14_3.41351(e-5)_0.27236_0.08919_276_276_  and global_step = 11750 (-750) 0.94078124
  #TODO monitor discr loss. Try raising lambda during training.

  #TODO non adv Current minimum: 0.2001 PARAMS [12, 4.5954111903935344e-05] LOSS 0.200081
  #TODO NON ADV Current minimum: 0.2027 PARAMS [14, 1.6419103411645864e-05] LOSS 0.2026753
  print("PARAMS", hp_tuning_result.x, "LOSS", hp_tuning_result.fun)
  #results = hyperparams_wrapper([8, 0.00012212957106648347, 0.051983923229111814, 0.0483198888684224])
  #print(results)

if __name__ == "__main__":
  #flags.mark_flag_as_required("data_dir")
  #flags.mark_flag_as_required("task_name")
  #flags.mark_flag_as_required("vocab_file")
  #flags.mark_flag_as_required("bert_config_file")
  #flags.mark_flag_as_required("output_dir")
  tf.app.run()