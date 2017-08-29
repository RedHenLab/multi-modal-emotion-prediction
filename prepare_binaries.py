import re
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from random import shuffle

# for Word2Vec embeddings
import gensim
model = gensim.models.Word2Vec.load_word2vec_format(DATA_PATH+'GoogleNews-vectors-negative300.bin', binary=True)  

DATA_PATH = '/home/karolina/Documents/GSOC/data/'
# file names are *.waw_st.npy
numpy_features = glob.glob(DATA_PATH+'IEMOCAP_full_release/Session*/sentences/wav/*/*st.npy')
# file names are *.wdseg
forced_alignments = glob.glob(DATA_PATH+'IEMOCAP_full_release/Session*/sentences/ForcedAlignment/Ses*/*.wdseg')
# *.txt files have assignments for the whole session not sentences
emo_evaluations = glob.glob(DATA_PATH+'IEMOCAP_full_release/Session*/dialog/EmoEvaluation/*.txt')

emotions_names = {'ang': np.int32(0), 'exc':np.int32(1), 'fru':np.int32(2), 
                  'hap':np.int32(3), 'neu':np.int32(4), 'sad':np.int32(5)}

# audio features dict - assigning paths to file names *(without extension)
features_dict = {}
for p in numpy_features:
    features_dict[p.split('/')[-1].split('.')[0]] = np.load(p)

# transcript files dict - assigning paths to file names *(without extension)    
transcripts_paths_dict = {}
for p in forced_alignments:
    transcripts_paths_dict[p.split('/')[-1].split('.')[0]] = p
    

###################### READ EMO FILES #####################

# messy evaluation dict - assigning evaluations (grand average) - from summary files to file names
file = emo_evaluations[1]
mean_emo_eval_dict = {}
emotional_eval_dict = {}

DATA_FLAG = False
values = []     # this is an empty value needed fot the loop
key = ''        # this as well
for file in emo_evaluations:
    with open(file) as f:
        for line in f:
            split = line.split('\t')
            if len(split)>1:
                split2 = split[1].split(';')
                if len(split2)==1:
                    if len(values)!=0:
                        real_values = np.array([float(re.sub("[^0-9.]"," ", v)) for v in values.split(' ')
                                               ], dtype=np.float32)
                        mean_emo_eval_dict[key]=real_values
                        all_emotions.append(emo_tag)
                        if emo_tag in emotions_names:
                            emotional_eval_dict[key] = emotions_names[emo_tag]
                    key = split2[0]
                    emo_tag = split[2]
                    values = split[3]

real_values = np.array([float(re.sub("[^0-9.]"," ", v)) for v in values.split(' ')], dtype=np.float32)
mean_emo_eval_dict[key]=real_values #appending the last evaluation
if emo_tag in emotions_names:
    emotional_eval_dict[key] = emotions_names[emo_tag]

###################### READ TRANSCRIPTS AND EMBEDDINGS #####################

LEN_WORD_LIM = 60
NUM_FEATURES = 34
LEN_SENTENCE = 25

def read_transcript(path):
    """
    Reads and cleans the transcript from the forced-alignment files.
    Cleans the 'non-words'
    """
    df = pd.read_csv(path,delim_whitespace=True, usecols=['SFrm','EFrm','Word']).dropna()
    df = df[~df.Word.str.contains('<')]
    df = df[~df.Word.str.contains('"')]
    df = df[~df.Word.str.contains('LAUGHTER')]
    df = df[~df.Word.str.contains('BREATHING')]
    df = df[~df.Word.str.contains('GARBAGE')]
    n_rows = df.shape[0]
    return df, n_rows

def clean_word(word):
    word = re.sub("[^a-zA-Z]","", word).lower()
    return word

def pad_feature(feature,audio_start,audio_end):
    empty_feature = np.zeros((NUM_FEATURES,LEN_WORD_LIM), dtype=np.float32) #all features have this size
    empty_feature[:,:audio_end-1-audio_start] = feature[:,audio_start:audio_end-1]
    audio_feature = empty_feature[:,:,None]
    return audio_feature

def pad_feature_sequence(feature_sequence):
    sequence = np.zeros((NUM_FEATURES,LEN_WORD_LIM,LEN_SENTENCE),dtype=np.float32)
    sequence[:,:,:feature_sequence.shape[2]] = feature_sequence
    return sequence

def pad_word_sequence(word_sequence):
    sequence = np.zeros((300,LEN_SENTENCE),dtype=np.float32)
    sequence[:,:word_sequence.shape[1]] = word_sequence
    return sequence

ready_audio_dict = {}
ready_word_embed_dict = {}
sentences_len_dict = {}

all_words = []
w2v_words = []
short_words = []
rejected_words = []

ALL_KEYS = emotional_eval_dict.keys() #transcripts_paths_dict.keys()
for key in list(ALL_KEYS):
    features = features_dict[key]
    if key in transcripts_paths_dict:
        transcript, n_rows = read_transcript(transcripts_paths_dict[key])
        OUTPUT_FLAG = False
        for n in range(n_rows): #iteration over the whole sentence
            w = transcript.Word.iloc[n]
            lower = clean_word(w) #cleaning up the transcript
            audio_start = int(transcript.SFrm.iloc[n])
            audio_end = int(transcript.EFrm.iloc[n])
            if audio_end-audio_start<LEN_WORD_LIM:
                if lower in model.vocab:
                    word_embed = model[lower][:,None]
                else:
                    word_embed = np.zeros((300,1))
                feature = features_dict[key]
                audio_feature = pad_feature(feature,audio_start,audio_end)
                if OUTPUT_FLAG:
                    audio = np.concatenate((audio,audio_feature),axis=2)
                    word = np.concatenate((word,word_embed),axis=1)
                else:
                    audio = audio_feature
                    word = word_embed
                    OUTPUT_FLAG = True
        if word.shape[1] < LEN_SENTENCE:          
            ready_audio_dict[key] = pad_feature_sequence(audio)
            ready_word_embed_dict[key] = pad_word_sequence(word)
            sentences_len_dict[key] = np.int32(audio.shape[-1])
del model
GOOD_KEYS = ready_audio_dict.keys()

###################### DIVIDE DATA INTO SPLITS #####################

hash_dictionary = {}
for sess_name in list(GOOD_KEYS):
    split = sess_name.split('_')
    ses_num = split[0][3:5]
    ses_g = split[0][-1]
    scenario = re.sub("[^a-z]"," ", split[1]).strip(' \t\n\r')
    scenario_num = re.sub("[^0-9]"," ", split[1]).strip(' \t\n\r')
    if len(split)==4:
        division_num = split[2]
        g = split[3][0]
    else:
        division_num = '0'
        g = split[2][0]
    hash_key = ses_num+ses_g+scenario+scenario_num+division_num+g
    if hash_key in hash_dictionary:
        hash_dictionary[hash_key].append(sess_name)
    else:
        hash_dictionary[hash_key] = [sess_name]

n_splits = 8
splits = np.empty(n_splits,dtype=object)

EMPTY_FLAG = True
for k, v in hash_dictionary.items():
    a = np.arange(len(v))
    shuffle(v)
    if EMPTY_FLAG:
        for i in range(n_splits):
            splits[i] = list(np.array(v)[a%n_splits==i])
        EMPTY_FLAG = False
    else:
        for i in range(n_splits):
            splits[i].extend(list(np.array(v)[a%n_splits==i]))      


######################	WRITE TO BINARIES #####################

BINARIES_PATH = '/home/karolina/Documents/GSOC/multi-modal-emotion-prediction/data_IEMOCAP/'

for j,split in enumerate(list(splits)):
    for i,key in enumerate(split):

        # dividing into files
        if i%100==0:
            train_filename = BINARIES_PATH+'split_'+str(j)+'_'+str(i//100)+'.tfrecords'

        writer = tf.python_io.TFRecordWriter(train_filename)
        
        audio = ready_audio_dict[key].tobytes()
        sentence_len = sentences_len_dict[key].tobytes()
        word = ready_word_embed_dict[key].tobytes()
        emo = mean_emo_eval_dict[key].tobytes()
        label = emotional_eval_dict[key].tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'audio_features'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio])),
            'sentence_len'      : tf.train.Feature(bytes_list=tf.train.BytesList(value=[sentence_len])),
            'word_embeddings'   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[word])),
            'y'                 : tf.train.Feature(bytes_list=tf.train.BytesList(value=[emo])),
            'label'             : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            }))

        writer.write(example.SerializeToString())
        writer.close()

