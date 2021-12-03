# <h1 align="center">:star2: Automatique Speech Recognition in Wolof :star2:</h1>

Automatic Speech Recognition or ASR is one of the tasks in NLP which consists of transcribing the corresponding text onto an audio clip. 

With the advent of deep learning, significant advances have been made in terms of speech recognition.

In this repository, we will implement the models that have allowed this advance in the Wolof language.  



## Objective

To achieve our goals in this project we will implement two models related to the paper:

* \[2015/12\] [**Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**](https://arxiv.org/abs/1512.02595)  
  

* \[2015/08\] [**Listen, Attend and Spell**](https://arxiv.org/abs/1508.01211)  
  
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

 
install the dependencies for this project by running the following commands in your terminal:

```
 pip install -r requirements.txt
```

run the deepspeech2 model by running the following command in your terminal:

```
python deep-specch2/src/train.py --train_file="./input/Train.csv" \
                        --dev_file="./input/Test.csv" \
                        --audio_dir="./input/clips" \
                        --n_filters=256 \
                        --conv_stide=2 \
                        --conv_border='valid' \
                        --n_lstm_units=256 \
                        --n_dense_units=42 \
                        --epochs=10 \
                        --batch_size=32 \
                        --output_dir="./output" \
```

## Ressources

Here are some useful papers for automatique speech recognition :
  
* \[2012/11\] [**Sequence Transduction with Recurrent Neural Networks**](https://arxiv.org/abs/1211.3711)   
  
* \[2014/11\] [**Voice Recognition Using MFCC Algorithm**](https://www.ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf)   
  
* \[2014/12\] [**Deep Speech: Scaling up end-to-end speech recognition**](https://arxiv.org/abs/1412.5567)  
  
* \[2015/06\] [**Attention-Based Models for Speech-Recognition**](https://arxiv.org/abs/1506.07503)  
  
* \[2015/08\] [**Listen, Attend and Spell**](https://arxiv.org/abs/1508.01211)  
  
* \[2015/12\] [**Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**](https://arxiv.org/abs/1512.02595)  
  
* \[2017/06\] [**Advances in Joint CTC-Attention based E2E ASR with a Deep CNN Encoder and RNN-LM**](https://arxiv.org/abs/1706.02737)   
  
* \[2017/07\] [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)   
  
* \[2017/12\] [**State-of-the-art Speech Recognition with Sequence-to-Sequence Models**](https://arxiv.org/abs/1712.01769) 
  
* \[2017/12\] [**An Analsis Of Incorporating An External Language Model Into A Sequence-to-Sequence Model**](https://arxiv.org/abs/1712.01996)   
  
* \[2018/04\] [**Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition**](https://ieeexplore.ieee.org/document/8462506)  
  
* \[2019/02\] [**On the Choice of Modeling Unit for Sequence-to-Sequence Speech Recognition**](https://arxiv.org/abs/1902.01955)  
  
* \[2019/04\] [**SpecAugment:  A Simple Data Augmentation Method for Automatic Speech Recognition**](https://arxiv.org/abs/1904.08779)  
  
* \[2019/04\] [**wav2vec: Unsupervised Pre-training for Speech Recognition**](https://arxiv.org/abs/1904.05862?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)  
  
* \[2019/05\] [**Transformers with convolutional context for ASR**](https://arxiv.org/abs/1904.11660)  
  
* \[2019/08\] [**Korean Grapheme Unit-based Speech Recognition Using Attention-CTC Ensemble Network**](https://ieeexplore.ieee.org/abstract/document/8836146)  
  
* \[2019/08\] [**Jasper: An End-to-End Convolutional Neural Acoustic Model**](https://arxiv.org/pdf/1904.03288.pdf)  
  
* \[2019/11\] [**End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures**](https://arxiv.org/abs/1911.08460)  
  
* \[2019/12\] [**SpecAugment on Large Scale Datasets**](https://arxiv.org/abs/1912.05533)  
   
* \[2020/04\] [**ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for ASR of Contact Centers**](https://arxiv.org/abs/2004.09367)  
  
* \[2020/05\] [**ContextNet: Improving Convolutional Neural Networks for ASR with Global Context**](https://arxiv.org/abs/2005.03191)  
  
* \[2020/05\] [**Conformer: Convolution-augmented Transformer for Speech Recognition**](https://arxiv.org/abs/2005.08100)  
  
* \[2020/06\] [**wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**](https://arxiv.org/abs/2006.11477)  
