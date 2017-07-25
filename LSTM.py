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

######################## FLAGS ########################

RUN = 'run_name'

# paths to tf binaries + splitting into validation and test set

HOME_PATH = '/home/karolina/Documents/GSOC'
RECORD_FILES = glob.glob(HOME_PATH+'/data/audio_features/IEMOCUP/*')
VALIDATION_SPLIT = glob.glob(HOME_PATH+'/data/audio_features/IEMOCUP/*_7_*')
TRAIN_SPLIT = list(set(RECORD_FILES) - set(VALIDATION_SPLIT))

# path where train logs will be saved

LOGDIR = HOME_PATH+'/GSOC/training_logs/'+RUN+'/'

# constants and flags 

Y_SHAPE = 3
N_LABELS = 7
N_FEATURES = 34
N_WORDS = 18
LEN_WORD_FEATURES = 240
EMBEDDING_SIZE = 300
BATCH_SIZE = 100
WORD_LSTM_REUSE = False
CELL_SIZE = 5
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
                            'word_embeddings'   : tf.FixedLenFeature([],tf.string),
                            'y'                 : tf.FixedLenFeature([],tf.string),
                            'label'             : tf.FixedLenFeature([],tf.string),
                                    },  name='tf_features')
    audio_features = tf.decode_raw(tfrecord_features['audio_features'],tf.float32)
    audio_features = tf.reshape(audio_features, (N_FEATURES,LEN_WORD_FEATURES,N_WORDS))
    audio_features.set_shape((N_FEATURES,LEN_WORD_FEATURES,N_WORDS))
    
    word_embeddings = tf.decode_raw(tfrecord_features['word_embeddings'],tf.float32)
    word_embeddings = tf.reshape(word_embeddings, (EMBEDDING_SIZE,N_WORDS))
    word_embeddings.set_shape((EMBEDDING_SIZE,N_WORDS))
    
    y = tf.decode_raw(tfrecord_features['y'],tf.float32)
    y.set_shape((Y_SHAPE))
    
    label = tf.decode_raw(tfrecord_features['label'],tf.int32)
    label.set_shape((1,))
    
    return audio_features, word_embeddings, y, label

def word_batch_splitter(audio_input):
    """
    Splits and flattens audio features within an audio-word.
    """
    N_SPLITS = 24
    split_audio = tf.split(audio_input, num_or_size_splits=N_SPLITS, axis=2)
    shape = split_audio[0].get_shape().as_list()
    split_audio = [tf.reshape(tf.squeeze(tensor), [shape[0],shape[1]*shape[2]]) for tensor in split_audio]
    return split_audio

def sequence_batch_splitter(inputs_sequence_batch):
    """
    Splits sentences.
    """
    N_SPLITS = 18
    ax = len(inputs_sequence_batch.get_shape().as_list())-1
    splitted_inputs = tf.split(inputs_sequence_batch, num_or_size_splits=N_SPLITS, axis=ax)
    return splitted_inputs

def produce_input_sequence(audio, words):
    """
    Concatenates corresponding word2vec embeddings with output of LSTM for audio words.
    """
    split_words = sequence_batch_splitter(words)
    sentences = [tf.concat([audio[i], tf.squeeze(split_words[i])], axis=1) for i in range(N_WORDS)]
    return sentences

def init_LSTM(size):
    """
    Initialization of LSTM cell. 
    TO DO's: unfolding, forget biasses=1, perhaps writing the cell by hand and tuning the initializers
    """
    rnn_cell = rnn.LSTMCell(size,initializer=tf.contrib.layers.xavier_initializer())
    return rnn_cell
     
def word_LSTM(splitted_inputs,lstm_fw_cell, lstm_bw_cell):
    """
    Runs LSTM over chunks of the word. The global var is needed for correct reusing.
    """
    global WORD_LSTM_REUSE
    with tf.variable_scope("audio_word_lstm") as scope:
        if not WORD_LSTM_REUSE:
            outputs = run_LSTM(lstm_fw_cell, lstm_bw_cell, splitted_inputs)
            WORD_LSTM_REUSE = True
        else:
            scope.reuse_variables()
            outputs = run_LSTM(lstm_fw_cell, lstm_bw_cell, splitted_inputs)
    return outputs

def run_LSTM(lstm_fw_cell, lstm_bw_cell, splitted_inputs):
    """
    Runs and returns the right output of the cell
    """
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, splitted_inputs, dtype=tf.float32)
    return outputs[-1]

def regression_layer(lstm_output):
    """
    Projects to a prediction vector of size N_LABEL.
    """
    shape = lstm_output.get_shape().as_list()
    net = tf.layers.dense(lstm_output, units=N_LABELS, name='regression')
    return net

def model(audio, words, lstm_fw_cell_1, lstm_bw_cell_1, lstm_fw_cell_2, lstm_bw_cell_2, reuse=False):
    with tf.variable_scope("model") as scope:
        if reuse:
            scope.reuse_variables()
        # audio - first LSTM
        audio_sentence = sequence_batch_splitter(audio)
        audio_chunks_sentence = [word_batch_splitter(audio) for audio in audio_sentence]
        audio_lstm = [word_LSTM(audio,lstm_fw_cell_1, lstm_bw_cell_1) for audio in audio_chunks_sentence]
        
        # audio + word - second LSTM
        sentence_in = produce_input_sequence(audio_lstm, words)
        sentence_out = run_LSTM(lstm_fw_cell_2, lstm_bw_cell_2,sentence_in)
        prediction = regression_layer(sentence_out)
    return prediction

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
    audio_feature, word_embedding, _, label = read_from_tfrecord(TRAIN_SPLIT)
    audio_features, word_embeddings, labels = tf.train.batch([audio_feature, word_embedding, label], 
                                                            batch_size=BATCH_SIZE, capacity=256, num_threads=15)

    test_audio_feature, test_word_embedding, _, test_label = read_from_tfrecord(VALIDATION_SPLIT)
    test_audio_features, test_word_embeddings, test_labels = tf.train.batch([test_audio_feature, 
                                                                test_word_embedding, test_label], 
                                                                batch_size=BATCH_SIZE, capacity=256, num_threads=15)
    lstm_fw_cell_1 = init_LSTM(CELL_SIZE)
    lstm_bw_cell_1 = init_LSTM(CELL_SIZE)
    lstm_fw_cell_2 = init_LSTM(CELL_SIZE)
    lstm_bw_cell_2 = init_LSTM(CELL_SIZE)

    predictions = model(audio_features, word_embeddings,
                       lstm_fw_cell_1, lstm_bw_cell_1,
                       lstm_fw_cell_2,lstm_bw_cell_2)

    test_predictions = model(test_audio_features,test_word_embeddings,
                            lstm_fw_cell_1, lstm_bw_cell_1,
                            lstm_fw_cell_2, lstm_bw_cell_2, reuse=True)


    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,predictions)
    tf.summary.scalar('loss', cross_entropy)

    test_cross_entropy = tf.losses.sparse_softmax_cross_entropy(test_labels,test_predictions)
    tf.summary.scalar('test_loss', cross_entropy)

    summary_accuracy(predictions, tf.squeeze(labels),'accuracy')    
    summary_accuracy(test_predictions, tf.squeeze(test_labels),'test_accuracy')  

    global_step = slim.get_or_create_global_step()

    # for SGD with momentum:
    learning_rate = tf.train.exponential_decay(
                       0.01,                      # Base learning rate.
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

