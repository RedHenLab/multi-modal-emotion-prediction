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

# set a constant random seed for comparable results

import random 
random.seed(0)

######################## FLAGS ########################

# paths to tf binaries + splitting into validation and test set

#HOME_PATH = '/home/karolina/Documents/GSOC'
RECORD_FILES = glob.glob('data_IEMOCAP/*')
VALIDATION_SPLIT = glob.glob('data_IEMOCAP/*_7_*')
TRAIN_SPLIT = list(set(RECORD_FILES) - set(VALIDATION_SPLIT))

# constants and flags 

Y_SHAPE = 3
N_LABELS = 6
N_FEATURES = 34
LEN_SENTENCE = 25
LEN_WORD = 60
EMBEDDING_SIZE = 300
BATCH_SIZE = 32
WORD_LSTM_REUSE = False
N_HIDDEN = 32
N_HIDDEN_2 = 16
LEARNING_RATE = 0.0001
EPOCH = int(5500/BATCH_SIZE)
STEPS = 60*EPOCH
DECAY = 20*EPOCH
DECAY_RATE = 0.5

# run name

RUN = 'multimodal_wlen'+str(LEN_WORD)+'_slen'+str(LEN_SENTENCE)+'_batchsize'+str(BATCH_SIZE)+'_bilstm'+str(N_HIDDEN)+'/'+str(N_HIDDEN_2)+'_learning_rate'+str(LEARNING_RATE)+'_dropout0.5'

# path where train logs will be saved

LOGDIR = 'training_logs/'+RUN+'/'

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
                            'sentence_len'      : tf.FixedLenFeature([],tf.string),
                            'word_embeddings'   : tf.FixedLenFeature([],tf.string),
                            'y'                 : tf.FixedLenFeature([],tf.string),
                            'label'             : tf.FixedLenFeature([],tf.string),
                                    },  name='tf_features')
    audio_features = tf.decode_raw(tfrecord_features['audio_features'],tf.float32)
    audio_features = tf.reshape(audio_features, (N_FEATURES,LEN_WORD,LEN_SENTENCE))
    audio_features.set_shape((N_FEATURES,LEN_WORD,LEN_SENTENCE))
    
    word_embeddings = tf.decode_raw(tfrecord_features['word_embeddings'],tf.float32)
    word_embeddings = tf.reshape(word_embeddings, (EMBEDDING_SIZE,LEN_SENTENCE))
    word_embeddings.set_shape((EMBEDDING_SIZE,LEN_SENTENCE))
    
    y = tf.decode_raw(tfrecord_features['y'],tf.float32)
    y.set_shape((Y_SHAPE))
    
    label = tf.decode_raw(tfrecord_features['label'],tf.int32)
    label.set_shape((1,))
    
    sentence_len = tf.decode_raw(tfrecord_features['sentence_len'],tf.int32)
    sentence_len.set_shape((1,))
    
    return audio_features, word_embeddings, label, sentence_len 

def init_LSTM(size):
    rnn_cell = rnn.LSTMCell(size,initializer=tf.contrib.layers.xavier_initializer())
    return rnn_cell
     
def word_LSTM(lstm_fw_cell, lstm_bw_cell, inputs, time_steps=LEN_WORD):
    global WORD_LSTM_REUSE
    with tf.variable_scope("audio_word_lstm") as scope:
        if not WORD_LSTM_REUSE:
            outputs = bidirectional_dyn_rnn(lstm_fw_cell, lstm_bw_cell, inputs, time_steps)
            WORD_LSTM_REUSE = True
        else:
            scope.reuse_variables()
            outputs = bidirectional_dyn_rnn(lstm_fw_cell, lstm_bw_cell, inputs, time_steps)
    return outputs

def bidirectional_dyn_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, time_steps, **kwargs):
    if 's_len' in kwargs:
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs,
                                                     sequence_length=kwargs['s_len'], dtype=tf.float32)
    else:
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs,
                                                     dtype=tf.float32)
    out = tf.concat([tf.squeeze(tf.split(o, num_or_size_splits=time_steps, axis=1)[-1]) for o in outputs],axis=1)
    return out

def regression_layer(lstm_output,reuse=False):
    with tf.variable_scope("regression") as scope:
        if reuse:
            scope.reuse_variables()
        shape = lstm_output.get_shape().as_list()
        net = tf.layers.dense(lstm_output, units=N_LABELS)
    return net

def audio_LSTM(audio_features, sentence_len,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                reuse=False):
    with tf.variable_scope("lstm_audio") as scope:
        if reuse:
            scope.reuse_variables()
        features = tf.split(audio_features, num_or_size_splits=BATCH_SIZE, axis=0)
        features = [tf.transpose(tf.squeeze(f), perm=[2, 1, 0]) for f in features]
        features = [tf.layers.dropout(f,0.5) for f in features]
        lstm_1 = [word_LSTM(lstm_fw_cell_1, lstm_bw_cell_1, f) for f in features]
        lstm_1 = tf.stack(lstm_1,0)
        lstm_1 = tf.layers.dropout(lstm_1,0.5)
        lstm_2 = bidirectional_dyn_rnn(lstm_fw_cell_2, lstm_bw_cell_2, 
                                       lstm_1, LEN_SENTENCE)#, s_len=tf.squeeze(sentence_len))
    return lstm_2

def words_LSTM(word_embeddings,
                lstm_fw_cell, 
                lstm_bw_cell,
                reuse=False):
    with tf.variable_scope("lstm_words") as scope:
        if reuse:
            scope.reuse_variables()
        word_embeddings = tf.transpose(word_embeddings, perm=[0,2,1])
        word_embeddings = tf.layers.dropout(word_embeddings,0.5)
        lstm = bidirectional_dyn_rnn(lstm_fw_cell, lstm_bw_cell, word_embeddings, LEN_SENTENCE)
        return lstm
    
def combine_LSTM(audio_lstm,word_lstm,reuse=False):
    inputs = tf.concat([audio_lstm,word_lstm],axis=1)
    inputs = tf.layers.dropout(inputs,0.5)
    reg = regression_layer(inputs,reuse=reuse)
    return reg

def multimodal_LSTM(word_embeddings,
                lstm_fw_cell, 
                lstm_bw_cell,
                audio_features, 
                sentence_len,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                reuse=False
                   ):
    audio = audio_LSTM(audio_features, 
                sentence_len,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                reuse=reuse)
    words = words_LSTM(word_embeddings,
                lstm_fw_cell, 
                lstm_bw_cell,
                reuse=reuse)
    combined = combine_LSTM(audio,words,reuse)
    return combined


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

    # Reading data

    audio_feature, word_embedding, label, sentence_len = read_from_tfrecord(TRAIN_SPLIT)
    audio_features, word_embeddings, labels, sentence_lens = tf.train.shuffle_batch([audio_feature, 
                                                                    word_embedding, 
                                                                    label, 
                                                                    sentence_len],
                                                                    batch_size=BATCH_SIZE, 
                                                                    capacity=256, num_threads=15,
                                                                    min_after_dequeue=200)

    test_audio_feature, test_word_embedding, test_label,test_sentence_len = read_from_tfrecord(VALIDATION_SPLIT)
    test_audio_features, test_word_embeddings, test_labels, test_sentence_lens = tf.train.shuffle_batch([test_audio_feature, 
                                                                                         test_word_embedding,  
                                                                                         test_label,
                                                                                         test_sentence_len], 
                                                                                         batch_size=BATCH_SIZE, 
                                                                                         capacity=256, num_threads=15,
                                                                                         min_after_dequeue=200)
    # Initializing model

    lstm_fw_cell = init_LSTM(N_HIDDEN_2)
    lstm_bw_cell = init_LSTM(N_HIDDEN_2)
    lstm_fw_cell_1 = init_LSTM(N_HIDDEN)
    lstm_bw_cell_1 = init_LSTM(N_HIDDEN)
    lstm_fw_cell_2 = init_LSTM(N_HIDDEN_2)
    lstm_bw_cell_2 = init_LSTM(N_HIDDEN_2)

    predictions = multimodal_LSTM(word_embeddings,
                lstm_fw_cell, 
                lstm_bw_cell,
                audio_features, 
                sentence_len,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                reuse=False)

    test_predictions = multimodal_LSTM(word_embeddings,
                lstm_fw_cell, 
                lstm_bw_cell,
                audio_features, 
                sentence_len,
                lstm_fw_cell_1, 
                lstm_bw_cell_1,
                lstm_fw_cell_2, 
                lstm_bw_cell_2,
                reuse=True)
    
    # Loss and summaries

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,predictions)
    tf.summary.scalar('loss', cross_entropy)

    test_cross_entropy = tf.losses.sparse_softmax_cross_entropy(test_labels,test_predictions)
    tf.summary.scalar('test_loss', test_cross_entropy)

    summary_accuracy(predictions, tf.squeeze(labels),'accuracy')    
    summary_accuracy(test_predictions, tf.squeeze(test_labels),'test_accuracy')  

    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
                       LEARNING_RATE,             # Base learning rate.
                       global_step * BATCH_SIZE,  # Current index into the dataset.
                       DECAY * BATCH_SIZE,        # Decay step.
                       DECAY_RATE,                # Decay rate.
                       staircase=True)
    
    tf.summary.scalar('learning_rate', learning_rate)

    # for SGD with momentum:
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  # Second argument - momentum

    # for Adam:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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

