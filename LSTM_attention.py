import numpy as np
import pandas as pd
import glob
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.contrib.layers
rnn = tf.contrib.rnn
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


######################## FLAGS ########################

RUN = 'run_batchsize20_bilstm32_learning_rate001'

# paths to tf binaries + splitting into validation and test set

HOME_PATH = '.'
RECORD_FILES = glob.glob(HOME_PATH+'/data/audio_features/IEMOCUP_2/*')
VALIDATION_SPLIT = glob.glob(HOME_PATH+'/data/audio_features/IEMOCUP_2/*_7_*')
TRAIN_SPLIT = list(set(RECORD_FILES) - set(VALIDATION_SPLIT))

# path where train logs will be saved

LOGDIR = HOME_PATH+'/GSOC/training_logs/'+RUN+'/'

# constants and flags 

Y_SHAPE = 3
N_LABELS = 6
N_FEATURES = 34
N_WORDS = 50 #MAX LEN OF SENTENCE
LEN_WORD_FEATURES = 240
EMBEDDING_SIZE = 300
BATCH_SIZE = 20
WORD_LSTM_REUSE = False
CELL_SIZE = 16
EPOCH = int(6031/BATCH_SIZE)
STEPS = 50*EPOCH

######################## FUNCTIONS ########################

def read_from_tfrecord(filenames):
    """
    Reads and reshapes binary files from IEMOCUP data.
    """
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'audio_features'    : tf.FixedLenFeature([],tf.string),
                            'audio_len'         : tf.FixedLenFeature([],tf.string),
                            'sentence_len'      : tf.FixedLenFeature([],tf.string),
                            #'word_embeddings'   : tf.FixedLenFeature([],tf.string),
                            'y'                 : tf.FixedLenFeature([],tf.string),
                            'label'             : tf.FixedLenFeature([],tf.string),
                                    },  name='tf_features')
    audio_features = tf.decode_raw(tfrecord_features['audio_features'],tf.float32)
    audio_features = tf.reshape(audio_features, (N_FEATURES,LEN_WORD_FEATURES,N_WORDS))
    audio_features.set_shape((N_FEATURES,LEN_WORD_FEATURES,N_WORDS))
    
    #word_embeddings = tf.decode_raw(tfrecord_features['word_embeddings'],tf.float32)
    #word_embeddings = tf.reshape(word_embeddings, (EMBEDDING_SIZE,N_WORDS))
    #word_embeddings.set_shape((EMBEDDING_SIZE,N_WORDS))
    
    y = tf.decode_raw(tfrecord_features['y'],tf.float32)
    y.set_shape((Y_SHAPE))
    
    label = tf.decode_raw(tfrecord_features['label'],tf.int32)
    label.set_shape((1,))
    
    audio_len = tf.decode_raw(tfrecord_features['audio_len'],tf.int32)
    audio_len.set_shape((N_WORDS))
    
    sentence_len = tf.decode_raw(tfrecord_features['sentence_len'],tf.int32)
    sentence_len.set_shape((1,))
    
    return audio_features, y, label, audio_len, sentence_len

def _init_attention(n_hidden,idd='word'):

    attention_task = tf.Variable(tf.zeros([1, n_hidden*2]),
        name=idd+'attention_vector')

    trans_weights = tf.Variable(tf.random_uniform([n_hidden*2, n_hidden*2], -1.0, 1.0),
            name=idd+'transformation_weights')

    trans_bias = tf.Variable(tf.zeros([n_hidden*2]), name=idd+'_trans_bias')

    return attention_task, trans_weights, trans_bias

def calculate_attention(embed, attention_task, trans_weights, trans_bias):

    embeddings_flat = tf.reshape(embed, [-1, CELL_SIZE*2])

    # Now calculate the attention-weight vector.

    # tanh transformation of embeddings
    keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
        trans_weights), trans_bias))

    # reshape the keys according to our embed vector
    keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [CELL_SIZE*2]]))

    # calculate score for each word embedding and take softmax on it

    scores = math_ops.reduce_sum(keys * attention_task, [2])
    alignments = nn_ops.softmax(scores)

    # expand aligments dimension so that we can multiply it with embed tensor
    alignments = array_ops.expand_dims(alignments,2)

    alignments = tf.identity(alignments, name='attention_weights')

    # generate context vector by making 
    context_vector = math_ops.reduce_sum(alignments * embed, [1])
#   context_vector = tf.nn.dropout(context_vector, self.keep_prob)
    return context_vector

def init_LSTM(size):
    rnn_cell = rnn.LSTMCell(size,initializer=tf.contrib.layers.xavier_initializer())
    return rnn_cell
     
def word_LSTM(lstm_fw_cell, lstm_bw_cell, inputs,s_len, time_steps=LEN_WORD_FEATURES):
    global WORD_LSTM_REUSE
    with tf.variable_scope("audio_word_lstm") as scope:
        if not WORD_LSTM_REUSE:
            outputs = bidirectional_dyn_rnn(lstm_fw_cell, lstm_bw_cell, inputs, s_len, time_steps)
            WORD_LSTM_REUSE = True
        else:
            scope.reuse_variables()
            outputs = bidirectional_dyn_rnn(lstm_fw_cell, lstm_bw_cell, inputs, s_len, time_steps)
    return outputs



def bidirectional_dyn_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, s_len, time_steps):
    (fw_outputs,bw_outputs),(fw_state,bw_state)= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs,
                                                 sequence_length=s_len, dtype=tf.float32)
    outputs = tf.concat((fw_outputs, bw_outputs), 2)
    return outputs

def regression_layer(lstm_output):
    shape = lstm_output.get_shape().as_list()
    net = tf.layers.dense(lstm_output, units=N_LABELS, name='regression')
    return net

def double_lstm_with_attention(audio_features, audio_len, sentence_len,
                #word_embeddings,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                attention_word,
                trans_weights_word,
                trans_bias_word,
                attention_sent,
                trans_weights_sent,
                trans_bias_sent,
                reuse=False):

    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        features = tf.split(audio_features, num_or_size_splits=BATCH_SIZE, axis=0)
        audio_lens = tf.split(audio_len, num_or_size_splits=BATCH_SIZE, axis=0)
        print(features[1])
        features = [tf.transpose(tf.squeeze(f), perm=[2, 1, 0]) for f in features]
        print(features[1])
        features = [tf.layers.dropout(f,0.3) for f in features]
        lstm_1 = [word_LSTM(lstm_fw_cell_1, lstm_bw_cell_1, 
                            features[i], tf.squeeze(audio_lens[i])) for i in range(BATCH_SIZE)]
        lstm_1_attention = [calculate_attention(embed, attention_word, trans_weights_word,
                            trans_bias_word) for embed in lstm_1]

        lstm_1_attention = tf.stack(lstm_1_attention,0)

        print(lstm_1_attention)
        #lstm_1_plus_embed = tf.concat([lstm_1,tf.transpose(word_embeddings, perm=[0,2,1])],axis=2)
        #print(lstm_1_plus_embed)
        lstm_1_attention = tf.layers.dropout(lstm_1_attention,0.4)
        lstm_2 = bidirectional_dyn_rnn(lstm_fw_cell_2, lstm_bw_cell_2, 
                                       lstm_1_attention, tf.squeeze(sentence_len), N_WORDS)
        lstm_2_attention = calculate_attention(lstm_2, attention_sent, trans_weights_sent,
                                               trans_bias_sent)
        print(lstm_2_attention)
        lstm_2 = tf.layers.dropout(lstm_2_attention,0.4)
        regression = regression_layer(lstm_2_attention)
        #regression = tf.layers.dropout(regression,0.5)
    return regression

def summary_accuracy(predictions,labels,summary_name):
    """
    Compute average accuracy over the batch and write a summary.
    """
    accuracy = tf.nn.in_top_k(predictions, labels, k=1, name=None)
    accuracy = tf.to_float(accuracy)
    accuracy = tf.reduce_mean(accuracy)
    tf.summary.scalar(summary_name, accuracy)


######################## MAIN ########################

if __name__ == "__main__":
    audio_feature, _, label, audio_len, sentence_len = read_from_tfrecord(TRAIN_SPLIT)
    audio_features,  labels, audio_lens,sentence_lens = tf.train.batch([audio_feature, 
                                                                     # word_embedding, 
                                                                    label, 
                                                                    audio_len,
                                                                    sentence_len], 
                                                            batch_size=BATCH_SIZE, capacity=256, num_threads=15)

    test_audio_feature, _, test_label, test_audio_len,test_sentence_len = read_from_tfrecord(VALIDATION_SPLIT)
    test_audio_features, test_labels, test_audio_lens,test_sentence_lens = tf.train.batch([test_audio_feature, 
                                                                                         # test_word_embedding, 
                                                                                          test_label,
                                                                                          test_audio_len,
                                                                                          test_sentence_len], 
                                                            batch_size=BATCH_SIZE, capacity=256, num_threads=15)
    lstm_fw_cell_1 = init_LSTM(CELL_SIZE)
    lstm_bw_cell_1 = init_LSTM(CELL_SIZE)
    lstm_fw_cell_2 = init_LSTM(CELL_SIZE)
    lstm_bw_cell_2 = init_LSTM(CELL_SIZE)

    attention_word, trans_weights_word, trans_bias_word = _init_attention(CELL_SIZE ,idd='word')
    attention_sent, trans_weights_sent, trans_bias_sent = _init_attention(CELL_SIZE ,idd='sent')


    predictions = double_lstm_with_attention(audio_features, audio_lens, sentence_lens, #word_embeddings,
                       lstm_fw_cell_1, lstm_bw_cell_1,
                       lstm_fw_cell_2,lstm_bw_cell_2,
                       attention_word, trans_weights_word,
                       trans_bias_word, attention_sent,
                       trans_weights_sent, trans_bias_sent)

    test_predictions = double_lstm_with_attention(test_audio_features, test_audio_lens, test_sentence_lens, #test_word_embeddings,
                            lstm_fw_cell_1, lstm_bw_cell_1,
                            lstm_fw_cell_2, lstm_bw_cell_2,
                            attention_word, trans_weights_word,
                            trans_bias_word, attention_sent,
                            trans_weights_sent, trans_bias_sent,
                            reuse=True)


    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,predictions)
    tf.summary.scalar('loss', cross_entropy)

    test_cross_entropy = tf.losses.sparse_softmax_cross_entropy(test_labels,test_predictions)
    tf.summary.scalar('test_loss', test_cross_entropy)

    summary_accuracy(predictions, tf.squeeze(labels),'accuracy')    
    summary_accuracy(test_predictions, tf.squeeze(test_labels),'test_accuracy')  

    global_step = slim.get_or_create_global_step()

    # for SGD with momentum:
    learning_rate = tf.train.exponential_decay(
                       0.001,                      # Base learning rate.
                       global_step * BATCH_SIZE,  # Current index into the dataset.
                       1500,                      # Decay step.
                       0.1,                       # Decay rate.
                       staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  # Second argument - momentum

    # for Adam:
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    train_op = slim.learning.create_train_op(cross_entropy, optimizer, global_step=global_step)

    init = tf.global_variables_initializer()
    var_list = slim.get_variables()
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)   # Keep 10 newest checkpoints

    final_loss = slim.learning.train(
                                     train_op,
                                     logdir=LOGDIR,
                                     number_of_steps=STEPS,
                                     save_summaries_secs=10,    # Save and log to tensorboard every 10 sec
                                     log_every_n_steps=10,
                                     save_interval_secs=60*5,   # Checkpoint every 5 minutes
                                     saver=saver)

