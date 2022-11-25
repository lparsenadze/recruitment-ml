from models import get_logger
from args import get_processing_args
#import matplotlib.pyplot as plt
from settings import *
import json
import re
from time import sleep
from IPython import embed
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def load_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset

def do_basic_eda(dataset):
    """Perform basic Exploratory Data Analysis (EDA) to select model hyperparameters."""    
    logger.info("Performing Basic EDA...")
    
    num_samples = len(dataset)
    num_positives = sum([d['austen'] == 1 for d in dataset])
    num_negatives = sum([d['austen'] == 0 for d in dataset])
    
    logger.info("-- Number of Samples:")
    logger.info(f"\tTotal: {num_samples}")
    logger.info(f"\tPositives: {num_positives}")
    logger.info(f"\tnegatives: {num_negatives}\n")
    sleep(0.1)
        
    lens = [len(re.sub(r"[^a-zA-Z0-9]", " ", d['text']).split()) for d in dataset]
    percentiles = [15,25,50, 75, 80, 90]    

    logger.info("-- Sequence Lengths:")
    logger.info(f"\tMax Sequence Length: {max(lens)}")
    logger.info(f"\tMin Sequence Length: {min(lens)}")
    logger.info(f"\tSeq. Length Percentiles:")
    for p in percentiles:
        logger.info(f"\t\t {p}-th percentile = {np.percentile(lens, p)}")    
    logger.info('\n')
    sleep(0.1)

    texts = [re.sub(r"[^a-zA-Z0-9]", " ", d['text']).split() for d in dataset]
    words = [w for text in texts for w in text]
    logger.info("-- Unique Tokens:")
    logger.info(f"\tCased: {len(set(words))}")
    logger.info(f'\tUncased: {len(set([w.lower() for w in words]))}')
    logger.info(f'\tLemmatized: {len(set([WordNetLemmatizer().lemmatize(w.lower()) for w in words]))}')
    logger.info(f'\tLemmatized Verbs: {len(set([WordNetLemmatizer().lemmatize(w.lower(), pos="v") for w in words]))}')
    logger.info(f'\tStemmed: {len(set([PorterStemmer().stem(w) for w in words]))}\n')

def preprocess(s, method="lemm_v"):
    assert method in ['cased', 'uncased', 'lemm', 'lemm_v', 'stemm']
    s = s.strip()
    if method == 'cased':
        return s
    elif method == 'uncased':
        return s.lower()
    elif method == 'lemm_v':
        return WordNetLemmatizer().lemmatize(s.lower(), pos="v")
    elif method == 'lemm':
        return WordNetLemmatizer().lemmatize(s.lower())
    else:
        return PorterStemmer().stem(s.lower())

def get_vocab(dataset, method='lemm_v'):
    texts = [re.sub(r"[^a-zA-Z0-9]", " ", sample['text']).split() for sample in dataset]
    words = set([w for text in texts for w in text])
    words = set([preprocess(w, method=method) for w in words])
    vocab = dict(zip(words, range(2, len(words) +2)))
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab, {v: k for k, v in vocab.items()}

def get_X_y(dataset, vocab, method="lemm_v", max_seq_len=100):
    texts = [re.sub(r"[^a-zA-Z0-9]", " ", sample['text']).split() for sample in dataset]
    texts = [[preprocess(w, method=method) for w in text] for text in texts]
    X = [[vocab[w] for w in text] for text in texts]
    y = [sample['austen'] for sample in dataset]

    X = tf.keras.preprocessing.sequence.pad_sequences(maxlen=max_seq_len, 
        sequences=X, 
        padding="post", 
        value=vocab['<PAD>'])
 
    y = np.array(y)
    
    return X, y

def resample(X, y, strategy='UnderSample'):
    assert strategy in ['UnderSample', 'OverSample', "None"]
    if strategy == 'UnderSample':
        sampler = RandomUnderSampler()
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    elif strategy == 'OverSample':
        sampler = RandomOverSampler
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    else:
        return X, y

def save_vec_data(filepath, data):
    with open(filepath, 'wb') as f:
        np.savez(f, **data)    
    
if __name__ == "__main__":
    args_ = get_processing_args()
    logger = get_logger(filename=PROCESS_LOG_PATH)
    
    dataset = load_dataset(args_.dataset)    
    
    do_basic_eda(dataset)
   
    vocab, id2word = get_vocab(dataset)   
 
    logger.info('Vectorizing Dataset...')
    X, y = get_X_y(dataset, vocab, method=args_.method, max_seq_len=args_.max_seq_len)
    
    logger.info('\tBalancing class supports..')
    logger.info(f'\t\tOriginal class supports: pos - {y[y==1].shape[0]}; neg - {y[y==0].shape[0]}')
    X, y = resample(X, y, strategy=args_.strategy)
    logger.info(f'\t\tBalanced supports: pos - {y[y==1].shape[0]}; neg - {y[y==0].shape[0]}') 
    
    sleep(0.1)
    logger.info('\tSplitting datasets..')
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args_.train_test_split, random_state=RANDOM_SEED)
    
    sleep(0.1)
    logger.info("Saving vectorized datasets...")
    save_vec_data(args_.train_vecs, {'X': X_train, 'y': y_train})    
    save_vec_data(args_.test_vecs, {'X': X_test, 'y': y_test})    
    with open(args_.vocab_path, 'w') as f:
        json.dump(vocab, f)
    logger.info("Done.")

