# multi-modal-emotion-prediction

Project developed during Google Summer of Code.

Here a description what it's all about.

## Getting Started

The project consists of the data preprocessing part and a TensorFlow model. The simplest way to use it is downloading TensorFlow binaries and running the model part. Make sure to check that all the paths for data imports are correct.

```
here a short example of running from the command line.
```

The data pipeline is designed for the IEMOCAP data set, but you may use the function that writes to records with your custom data as long as it fits to the format.

### Prerequisites


#### Audio feature extraction (shell script)

- python 2.7 (there is a fair chance that pyAudioAnalysis would work with python 3+, but unfortunately the author haven't dared to try it out)
- pyAudioAnalysis + dependencies


#### Data pipeline

- python 3.6
- pandas
- TensorFlow
- gensim (for word2vec)

#### Model

- python 3.6
- TensorFlow



