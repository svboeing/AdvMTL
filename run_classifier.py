# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from statistics import mean

import collections
import csv
import os
import modeling
import optimization
import tokenization
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import preprocess_corpus
import numpy as np
from tensorflow.python.framework import ops
from skopt import gp_minimize

names = ['Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen',
         'Kindle_Store', 'Sports_and_Outdoors',
         'Cell_Phones_and_Accessories', 'Health_and_Personal_Care',
         'Toys_and_Games',
         'Video_Games', 'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android', 'Office_Products']

def diff_loss(A, B):
    return tf.norm(tf.matmul(tf.transpose(tf.to_float(A)), tf.to_float(B)))**2
#this is modified loss for rank 1

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

flags.DEFINE_bool(
    "adversarial", True,
    "Wherther to use encoders and adversarial module")

flags.DEFINE_float("orth_loss_weight", 0.1,
                   "Orth loss weight")

flags.DEFINE_float("discr_loss_weight", 0.05,
                   "Discr loss weight")

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

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 250,
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

  def __init__(self, guid, text_a_list, text_b=None, label_list=None):
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
    self.text_a_list = text_a_list
    self.text_b = text_b
    self.label_list = label_list


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
               input_ids_list,
               input_mask_list,
               segment_ids_list,
               label_id_list,
               is_real_example=True):
    self.input_ids_list = input_ids_list
    self.input_mask_list = input_mask_list
    self.segment_ids_list = segment_ids_list
    self.label_id_list = label_id_list
    self.is_real_example = is_real_example




def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, num_tasks):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids_list=[[0] * max_seq_length]*num_tasks,
        input_mask_list=[[0] * max_seq_length]*num_tasks,
        segment_ids_list=[[0] * max_seq_length]*num_tasks,
        label_id_list=[0]*num_tasks,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a_list = [tokenizer.tokenize(text) for text in example.text_a_list]
  assert len(tokens_a_list) == num_tasks
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    pass
  else:
    # Account for [CLS] and [SEP] with "- 2"
    #for tokens_a in tokens_a_list:
    #    if len(tokens_a) > max_seq_length - 2:
    #        tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens_a_list = [tokens_a[0:(max_seq_length - 2)] if len(tokens_a) > max_seq_length - 2 else tokens_a for tokens_a in tokens_a_list]
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
  tokens_list = []
  segment_ids_list = []
  #print(len(tokens_a_list))
  for tokens_a in tokens_a_list:
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    while len(segment_ids) < max_seq_length:
        segment_ids.append(0)
    assert len(segment_ids) == max_seq_length
    tokens_list.append(tokens)
    segment_ids_list.append(segment_ids)
  assert len(tokens_list) == len(segment_ids_list) == num_tasks
  '''if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)'''
  input_ids_list = []
  input_mask_list = []
  for tokens in tokens_list:
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
    input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        #segment_ids.append(0) done this already

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    #assert len(segment_ids) == max_seq_length
    input_ids_list.append(input_ids)
    input_mask_list.append(input_mask)
  assert len(input_ids_list) == len(input_mask_list) == num_tasks

  label_id_list = [label_map[label] for label in example.label_list]
  assert len(label_id_list) == num_tasks
  '''if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))'''

  feature = InputFeatures(
      input_ids_list=input_ids_list,
      input_mask_list=input_mask_list,
      segment_ids_list=segment_ids_list,
      label_id_list=label_id_list,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, num_tasks):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file) # OUTPUT FILE

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, num_tasks)

    def create_list_int_feature(values_list):
      assert len(values_list) == num_tasks
      values_list = np.asarray(values_list).flatten()
      assert len(values_list) == num_tasks*max_seq_length
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values_list))) #gotta reshape it back later!
      return f

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    #the hell is going on here

    features = collections.OrderedDict()
    features["input_ids_list"] = create_list_int_feature(feature.input_ids_list)
    features["input_mask_list"] = create_list_int_feature(feature.input_mask_list)
    features["segment_ids_list"] = create_list_int_feature(feature.segment_ids_list)
    features["label_ids_list"] = create_int_feature(feature.label_id_list) #labels_list are one-dimensional
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)]) #this remains the same

    tf_example = tf.train.Example(features=tf.train.Features(feature=features)) #type = int64list
    writer.write(tf_example.SerializeToString()) #serialize to binary string
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_tasks):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids_list": tf.FixedLenFeature([seq_length*num_tasks], tf.int64), #FLATTENED
      "input_mask_list": tf.FixedLenFeature([seq_length*num_tasks], tf.int64),#FLATTENED
      "segment_ids_list": tf.FixedLenFeature([seq_length*num_tasks], tf.int64),#FLATTENED
      "label_ids_list": tf.FixedLenFeature([num_tasks], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
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
          d = d.shuffle(buffer_size=10)

      d = d.apply(
          tf.contrib.data.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              drop_remainder=drop_remainder))

      return d

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


def create_model(bert_config, is_training, input_ids_list, input_mask_list, segment_ids_list,
                 labels_list, num_labels, use_one_hot_embeddings, num_tasks, seq_length):
  """Creates a classification model."""
  assert input_ids_list.shape.as_list()[1] == input_mask_list.shape.as_list()[1]\
         == segment_ids_list.shape.as_list()[1] == num_tasks*seq_length
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids_list=input_ids_list, # list
      input_mask_list=input_mask_list, # list
      token_type_ids_list=segment_ids_list, # list
      use_one_hot_embeddings=use_one_hot_embeddings,
      num_tasks = num_tasks,
      my_seq_length = seq_length
  )
  assert num_labels == 2

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer_list = model.get_pooled_output()
  print(len(output_layer_list))
  assert len(output_layer_list) == num_tasks
  loss_list = []
  per_example_loss_list = []
  logits_list, probabilities_list = [], []
  common_encoders_list = []
  hidden_size = output_layer_list[0].shape[-1].value

  orth_loss = 0

  for i in range(len(output_layer_list)):
    if is_training:
      # I.e., 0.1 dropout
      output_layer_list[i] = tf.nn.dropout(output_layer_list[i], keep_prob=0.9)

  for i in range(len(output_layer_list)):
      if FLAGS.adversarial:
            with tf.variable_scope("common_encoder", reuse=tf.AUTO_REUSE):
                common_encoders_list.append(tf.layers.dense(inputs=output_layer_list[i],
                                           units=hidden_size,
                                           activation=tf.nn.tanh,
                                           use_bias=True,
                                           name="common_encoder"
                                           )) #check that only one common encoder is created!

      with tf.variable_scope(str(i)+"_th_task"):
          assert hidden_size == output_layer_list[i].shape[-1].value

          if FLAGS.adversarial:
             output_weights = tf.get_variable(
                  "output_weights", [num_labels, 2*hidden_size],
                   initializer=tf.truncated_normal_initializer(stddev=0.02))
          else:
             output_weights = tf.get_variable(
                  "output_weights", [num_labels, hidden_size],
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

          output_bias = tf.get_variable(
              "output_bias", [num_labels], initializer=tf.zeros_initializer())

          if FLAGS.adversarial:
              private_encoder = tf.layers.dense(inputs=output_layer_list[i],
                                           units=hidden_size,
                                           activation=tf.nn.tanh,
                                           use_bias=True,
                                           )
              #orth_loss += diff_loss(tf.stack([common_encoders_list[-1], common_encoders_list[-1]]), tf.stack([private_encoder, private_encoder]))
              orth_loss += diff_loss(common_encoders_list[-1], private_encoder)
          with tf.variable_scope("loss"):
            if FLAGS.adversarial:
                logits_list.append(tf.matmul(tf.concat([common_encoders_list[-1],private_encoder], axis=-1), output_weights, transpose_b=True))
            else:
                logits_list.append(tf.matmul(output_layer_list[i], output_weights, transpose_b=True))
            logits_list[-1] = tf.nn.bias_add(logits_list[-1], output_bias)
            probabilities_list.append(tf.nn.softmax(logits_list[-1], axis=-1))
            log_probs = tf.nn.log_softmax(logits_list[-1], axis=-1)

            one_hot_labels = tf.one_hot(
                tf.transpose(labels_list)[:][i], depth=num_labels, dtype=tf.float32)

            per_example_loss_list.append(-tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
            loss_list.append(tf.reduce_mean(per_example_loss_list[-1])) #LOOOOOOOOOSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  all_common_encoded = tf.concat(common_encoders_list, axis=0) #COMMON_ENC_LIST[I] [BATCH, 768]
  #assert all_common_encoded.shape.as_list()[0] == num_tasks#batch_size here is strictly 1?
  #assert all_common_encoded.shape.as_list()[1] == hidden_size
  print(all_common_encoded.shape.as_list())
  if FLAGS.adversarial:
      with tf.variable_scope("task_discriminator"):  # ONLY ONE FOR ALL TASKS
          discriminator = tf.layers.dense(inputs=flip_gradient(all_common_encoded),
                                          units=len(names),
                                          # activation=tf.nn.tanh,
                                          use_bias=True,
                                          )
          discr_log_probs = tf.nn.log_softmax(discriminator, axis=-1)

          task_labels = []
          for i in range(num_tasks):
              #for j in range(int(all_common_encoded.shape.as_list()[0]/num_tasks)):
              for j in range(FLAGS.train_batch_size): #TODO fix this kostyl - what about eval batch_size
                   task_labels.append(tf.one_hot(i, num_tasks))


          discr_loss = tf.reduce_mean(-tf.reduce_sum(task_labels * discr_log_probs, axis=-1))
  else:
      discr_loss = 0




  assert len(loss_list) == len(per_example_loss_list) == len(logits_list) == len(probabilities_list) == num_tasks
  return (loss_list, per_example_loss_list, logits_list, probabilities_list, orth_loss, discr_loss) # whats with logits and probs? they go to metrics_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_tasks, seq_length):
  """Returns `model_fn` closure for TPUEstimator."""
  assert num_labels == 2
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids_list = features["input_ids_list"]#[num_tasks*seq_length] DECODING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    input_mask_list = features["input_mask_list"]#[num_tasks*seq_length]
    segment_ids_list = features["segment_ids_list"] #[num_tasks*seq_length]
    label_ids_list = features["label_ids_list"] # [num_tasks]
    is_real_example = None

    assert input_ids_list.shape.as_list()[1] == input_mask_list.shape.as_list()[1] == \
           segment_ids_list.shape.as_list()[1] == num_tasks*seq_length
    assert label_ids_list.shape.as_list()[1] == num_tasks

    assert "is_real_example" in features
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids_list), dtype=tf.float32) #????

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss_list, per_example_loss_list, logits_list, probabilities_list, orth_loss, discr_loss) = create_model(  # HO HO HO HERE IS THE LOSS
        bert_config, is_training, input_ids_list, input_mask_list, segment_ids_list, label_ids_list,
        num_labels, use_one_hot_embeddings, num_tasks, seq_length)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      total_loss = sum(total_loss_list) + FLAGS.discr_loss_weight * discr_loss + FLAGS.orth_loss_weight * orth_loss
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu) # GLOBAL OPTIMIZER

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss_list, label_ids_list, logits_list, is_real_example): #label_ids_list [batch, num_tasks]
        assert len(per_example_loss_list) == len(logits_list) == num_tasks
        assert  label_ids_list.shape.as_list()[1]==num_tasks
        label_ids_list = tf.transpose(label_ids_list) #[ num_tasks, batch]
        is_real_example_list, label_ids_list_task, predictions_list = [], [], []

        #for per_example_loss, label_ids, logits in zip(per_example_loss_list, label_ids_list, logits_list):
        for i in range(num_tasks):
            #per_example_loss, logits = per_example_loss_list[i], logits_list[i]
            label_ids_task = label_ids_list[:][i] #really??????? yes - checked on the jupyter
            label_ids_list_task.append(label_ids_task)
            is_real_example_list.append(is_real_example)
            argmaxed = tf.argmax(logits_list[i], axis=-1, output_type=tf.int32)
            print(argmaxed.shape.as_list())
            predictions_list.append(argmaxed)

        assert len(predictions_list) == num_tasks
        all_tasks_label_ids = tf.concat(label_ids_list_task, axis = 0)
        all_tasks_prediction = tf.concat(predictions_list, axis = 0)
        all_tasks_per_example_loss = tf.concat(per_example_loss_list, axis = 0)
        all_tasks_is_real_example = tf.concat(is_real_example_list, axis = 0)
        accuracy = tf.metrics.accuracy(
                labels=all_tasks_label_ids, predictions=all_tasks_prediction, weights=all_tasks_is_real_example)
        loss = tf.metrics.mean(values=all_tasks_per_example_loss, weights=all_tasks_is_real_example)
        return {
                "eval_accuracy": accuracy, # MEAN ACC ARE PASSED TO ESTIMATOR
                "eval_loss": loss, # MEAN LOSS IS PASSED TO ESTIMATOR
        }

      eval_metrics = (metric_fn, #assert len(label_ids_list) == num_tasks
                      [per_example_loss_list, label_ids_list, logits_list, is_real_example])
      total_loss = sum(total_loss_list)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    '''else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)'''
    return output_spec

  return model_fn



def _my_create_examples(texts_list, labels_list, num_tasks, corpus_length):
    examples = []

    for i in range(corpus_length): #iterate through dataset
        texts, labels = [], []
        for j in range(num_tasks): #iterate through each task
            texts.append(tokenization.convert_to_unicode(texts_list[j][i]).strip( '\n' ))
            labels.append(tokenization.convert_to_unicode(labels_list[j][i]).strip( '\n' ))

        examples.append(
                InputExample(guid=str(i), text_a_list=texts, label_list=labels))
    assert len(examples) == corpus_length
    return examples

def hyperparams_wrapper(params):
    tf.reset_default_graph()

    FLAGS.adversarial = True

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

    FLAGS.max_seq_length = 140
    FLAGS.train_batch_size = params[0]
    FLAGS.eval_batch_size = 1 #params[0]
    FLAGS.learning_rate = params[1]  # 2e-5
    FLAGS.orth_loss_weight = params[2]
    FLAGS.discr_loss_weight = params[3]
    FLAGS.common_enc_size = params[4]
    FLAGS.private_enc_size = params[5]
    FLAGS.num_train_epochs = 3.0
    FLAGS.output_dir = "./outputs/all_tasks_simult/hyperparams_search" + convert_params_to_string(params)

    tf.logging.set_verbosity(tf.logging.INFO)
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

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

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
    gpu_options = tf.GPUOptions(allow_growth=True)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=1,
        session_config=tf.ConfigProto(gpu_options=gpu_options), #report_tensor_allocations_upon_oom = True
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
        train_examples = _my_create_examples(texts_train_list, labels_train_list, len(names),
                                             corpus_length=train_size)  # list of examples
        eval_examples = _my_create_examples(texts_test_list, labels_test_list, len(names),
                                            test_size)  # list of examples
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
        num_tasks=len(names),
        seq_length=FLAGS.max_seq_length,
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
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file,
            len(names))  # put binary-serialized features to the train_file
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(  # THIS FUNCTION RETURNS TRAIN EXAMPLES AND LABELS
            input_file=train_file,  # HERE IS TRAIN FILE
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            num_tasks=len(names)
        )
        # okay SO HERE GOES THE TRAIN
        #estimator.train(input_fn=train_input_fn,
        #                max_steps=num_train_steps)  # that is for TPU but kinda falls back to normal estimator - what is it?

        stop_hook = tf.contrib.estimator.stop_if_no_decrease_hook(estimator, "eval_loss",
                                                                  max_steps_without_decrease=750, min_steps=750,
                                                                  run_every_secs=60)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[stop_hook])

    if FLAGS.do_eval:
        # eval_examples = processor.get_dev_examples(FLAGS.data_dir) # evaluating on dev - where is test?

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
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, num_tasks=len(names))

        tf.logging.info("***** Running evaluation *****")
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
            num_tasks=len(names)

        )

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=120,
                                          throttle_secs=0)
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
            # print("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                # print(key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return result["eval_loss"]

def main(_):
    hp_tuning_result = gp_minimize(hyperparams_wrapper, [[1,2], (1e-5, 1e-3), (1e-2, 0.5), (1e-3, 1e-1), [256, 424, 576, 768], [256, 424, 576, 768]],
                                   n_calls=30, x0=[2, 5.905854465935257e-05, 0.5, 0.0482198888684224, 424, 424],
                                   verbose=True)
    # 2_0.0001272644197_0.1275340837484_0.0637559622762_768_256_ is for 0.25
    #  TODO Current minimum: 0.1907 PARAMS [2, 5.905854465935257e-05, 0.5, 0.049983964758675954, 768, 576] LOSS 0.19069473
    #
    print("PARAMS", hp_tuning_result.x, "LOSS", hp_tuning_result.fun)


'''
  if FLAGS.do_predict: # say False
    predict_examples = processor.get_test_examples(FLAGS.data_dir) # predict goes on test
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples'''


if __name__ == "__main__":
  #flags.mark_flag_as_required("data_dir")
  #flags.mark_flag_as_required("task_name")
  #flags.mark_flag_as_required("vocab_file")
  #flags.mark_flag_as_required("bert_config_file")
  #flags.mark_flag_as_required("output_dir")
  tf.app.run()
