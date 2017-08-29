# multi-modal-emotion-prediction

Project developed during Google Summer of Code.
The repository contains a multi-modal machine learning model utilizing audio and text to prediction emotional rating of sentences. The model is based on bi-directional LSTM cells.
For more information visit the wiki page.

## Getting Started

The project consists of TensorFlow model and additional scripts for data preprocessing. 

### Running the training LSTM.py

####LSTM.py
This is the main file with model. Provided that you have cloned the repository, simply running this file from the command line will start the training of the network. If you want to change some hyperparameters - change them in this file.

### Data preprocessing
As the preprocessed data is already a part of the repository, the rest of the code doesn't need to be run on your pc unless you explicitly want to change something on the data preprocessing level. The data pipeline is designed for the IEMOCAP data set, but you may use the function that writes to records with your custom data as long as it fits to the format.

### data_preprocessing.sh
This is a console script for extracting audio features from .wav aufio files

### data_pipeline.py
This script makes TensorFlow records with ready training examples out of the data from IEMOCAP data set and audio features computed with data_preprocessing.sh.


### Prerequisites


#### Audio feature extraction (data_preprocessing.sh)

- python 2.7 (there is a fair chance that pyAudioAnalysis would work with python 3+, but unfortunately the author haven't dared to try it out)
- pyAudioAnalysis + dependencies


#### Data pipeline (data_pipeline.py)

- python 3.6
- pandas
- TensorFlow
- gensim (for word2vec)

#### Model (LSTM.py)

- python 3.6
- TensorFlow

