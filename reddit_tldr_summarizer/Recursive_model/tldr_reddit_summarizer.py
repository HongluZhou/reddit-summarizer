#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:07:30 2019

@author: zhouhonglu
"""

from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import json
import pdb
import re
from nltk.corpus import stopwords
import time
from datetime import datetime as dt
import logging
import pickle
import random
import string
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from langdetect import detect
import keras
import pandas as pd
from keras.layers import Input, CuDNNLSTM, RepeatVector, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras import backend as K


parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=1,
                    help='An integer: 1 to train, 2 to test')
args = parser.parse_args()
if args.run_opt == 1:
    run_opt = 'train'
else:
    run_opt = 'test'


def create_config():
    config = {
            'root_path': "/media/data1/summarizer/tldr",
            'exp_name': 'tldr_exp',

            'create_partition': True,
            'load_raw_data': True,
            'load_cleaned_data': False,
            'window': 10
            }

    return config


config = create_config()
root_path = config['root_path']
exp_path = os.path.join(root_path, config['exp_name'])
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
exp_data_path = os.path.join(root_path, config['exp_name'], 'data')
if not os.path.exists(exp_data_path):
    os.makedirs(exp_data_path)
if not os.path.exists(os.path.join(exp_data_path, 'dataset')):
    os.makedirs(os.path.join(exp_data_path, 'dataset'))


def set_logger(logger_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(logger_name, mode='w')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info(os.path.basename(__file__))
    logger.info(dt.now().strftime('%m/%d/%Y %I:%M:%S %p'))

    return logger


def print_and_log(logger, msg):
    print(msg)
    logger.info(msg)


logger = set_logger(os.path.join(
        exp_path, 'tldr_' + run_opt + '_' + dt.now().strftime(
                "%Y-%m-%dT%H-%M-%SZ") + '.log'))


if config['create_partition']:
    if config['load_raw_data']:
        data_path = os.path.join(root_path, "data", "tldr-training-data.jsonl")
        start_load = time.time()
        print_and_log(logger, 'start loading raw data...')
        with open(data_path, 'r') as f:
            raw_data = [json.loads(line) for line in f]
        print_and_log(logger, "loading raw data took {}".format(
                time.time()-start_load))
        """
        raw_data[0].keys()
        dict_keys(['author', 'body', 'content', 'content_len', 'id',
                   'normalizedBody', 'subreddit', 'subreddit_id', 'summary',
                   'summary_len'])
        """

    if not config['load_cleaned_data']:
        contractions = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he's": "he is",
                "how'd": "how did",
                "how'll": "how will",
                "how's": "how is",
                "i'd": "i would",
                "i'll": "i will",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it would",
                "it'll": "it will",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "must've": "must have",
                "mustn't": "must not",
                "needn't": "need not",
                "oughtn't": "ought not",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "she'd": "she would",
                "she'll": "she will",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "that'd": "that would",
                "that's": "that is",
                "there'd": "there had",
                "there's": "there is",
                "they'd": "they would",
                "they'll": "they will",
                "they're": "they are",
                "they've": "they have",
                "wasn't": "was not",
                "we'd": "we would",
                "we'll": "we will",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "where'd": "where did",
                "where's": "where is",
                "who'll": "who will",
                "who's": "who is",
                "won't": "will not",
                "wouldn't": "would not",
                "you'd": "you would",
                "you'll": "you will",
                "you're": "you are"
            }
        table = str.maketrans('', '', string.punctuation)
        stops = set(stopwords.words("english"))
        all_eng_words = set(words.words())
        all_eng_words.add('utc')
        porter = PorterStemmer()


def clean_text(text, remove_stopwords=True):
    '''Remove unwanted characters, stopwords, and format the text
    to create fewer nulls word embeddings'''
    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    text = re.split(r'\W+', text)
    # remove punctuation from each word
    text = [w.translate(table) for w in text]
    # remove all tokens that are not alphabetic
    text = [word for word in text if word.isalpha()]
    if remove_stopwords:
        # remove stop words
        text = [w for w in text if w not in stops]
    text = " ".join(text)
    # if non english word contained, lose the sample
    # stemmed = [porter.stem(word) for word in word_tokenize(text)]
    # for word in stemmed:
    #     if word not in all_eng_words:
    #         return ''
    try:
        text_lang_type = detect(text)
        if text_lang_type != 'en':
            return ''
    except:
        return ''
    return text


if config['create_partition']:
    if not config['load_cleaned_data']:
        start_clean = time.time()
        if not config['load_raw_data']:
            print("Please load raw data first!")
            os._exit(0)
        # clean data
        raw_data_new = dict()
        raw_data_new['text'] = []
        raw_data_new['summary'] = []
        cleaned_data = dict()
        cleaned_data['text'] = []
        cleaned_data['summary'] = []
        count_ignore = 0
        for i in range(len(raw_data)):
            print('cleaning data {}/{}  {} count_ignore: {} count_valid: {}'
                  ', alreaddy took {}'.format(
                    i+1, len(raw_data), round((i+1)/len(raw_data), 2),
                    count_ignore, len(cleaned_data['text']),
                    round(time.time()-start_clean, 2)))
            cleaned_text = clean_text(raw_data[i]['content'],
                                      remove_stopwords=False)
            cleaned_summary = clean_text(raw_data[i]['summary'],
                                         remove_stopwords=False)
            if len(cleaned_text) == 0 or len(cleaned_summary) == 0:
                count_ignore += 1
                continue
            cleaned_data['text'].append(cleaned_text)
            cleaned_data['summary'].append(cleaned_summary)
            raw_data_new['text'].append(raw_data[i]['content'])
            raw_data_new['text'].append(raw_data[i]['summary'])
            if len(cleaned_data['text']) > 100000:  # 300k
                break
        print_and_log(logger, "cleaning data took {}".format(
                time.time() - start_clean))
        with open(os.path.join(exp_data_path, "cleaned_data.pickle"),
                  'wb') as f:
            pickle.dump(cleaned_data, f)
        print('cleaned data saved!')
        raw_data = raw_data_new
        with open(os.path.join(exp_data_path, "raw_data.pickle"),
                  'wb') as f:
            pickle.dump(raw_data, f)
        print('raw data saved!')
    else:
        with open(os.path.join(root_path, "data", "cleaned_data.pickle"),
                  'rb') as f:
            cleaned_data = pickle.load(f)
        print('cleaned data loaded!')
        with open(os.path.join(root_path, "data", "raw_data.pickle"),
                  'rb') as f:
            raw_data = pickle.load(f)
        print('raw data loaded!')


def train_vali_test_creation():
    # check cleaned corpus
    print_and_log(logger, "checking cleaned corpus...")
    print_and_log(logger, "the number of text-summary pairs: {}".format(
            len(cleaned_data['text'])))

    all_word_corpus = set()
    text_word_corpus = set()
    summary_word_corpus = set()
    max_text_length = 0
    max_summary_length = 0
    for i in range(len(cleaned_data['text'])):
        print('checking cleaned corpus {}/{}  {}'.format(
                i+1, len(cleaned_data['text']),
                round((i+1)/len(cleaned_data['text']), 2)))
        tokenize_text = word_tokenize(cleaned_data['text'][i])
        tokenize_summary = word_tokenize(cleaned_data['summary'][i])
        max_text_length = max(max_text_length, len(tokenize_text))
        max_summary_length = max(max_summary_length, len(tokenize_summary))
        for word in tokenize_text:
            text_word_corpus.add(word)
            all_word_corpus.add(word)
        for word in tokenize_summary:
            summary_word_corpus.add(word)
            all_word_corpus.add(word)

    vocalbulary_all = len(all_word_corpus)
    vocalbulary_text = len(text_word_corpus)
    vocalbulary_summary = len(summary_word_corpus)

    print_and_log(logger, "the number of unique words totally: {}".format(
            vocalbulary_all))
    print_and_log(logger, "the number of unique words in text corpus: "
                  "{}".format(vocalbulary_text))
    print_and_log(logger, "the number of unique words in summary corpus: "
                  "{}".format(vocalbulary_summary))
    print_and_log(logger, "maximum length of words in text: "
                  "{}".format(max_text_length))
    print_and_log(logger, "maximum length of words in summary: "
                  "{}".format(max_summary_length))
    """
    the number of unique words totally: 1,252,589
    the number of unique words in text corpus: 1,161,606
    the number of unique words in summary corpus: 389,107
    """
    shuffled_index = list(range(len(cleaned_data['text'])))
    random.shuffle(shuffled_index)

    trainset_len = int(0.6*len(cleaned_data['text']))
    trainset_idx = shuffled_index[:trainset_len]
    valiset_len = int(0.3*len(cleaned_data['text']))
    valiset_idx = shuffled_index[trainset_len:trainset_len+valiset_len]
    testset_idx = shuffled_index[trainset_len+valiset_len:]
    testset_len = len(testset_idx)

    sample_idx = 0
    partition = dict()
    partition['train'] = []
    partition['validation'] = []
    labels = dict()

    all_word_corpus.add('<sos>')
    all_word_corpus.add('<eos>')
    all_word_corpus.add('<pad>')
    text_word_corpus.add('<sos>')
    text_word_corpus.add('<eos>')
    text_word_corpus.add('<pad>')
    summary_word_corpus.add('<sos>')
    summary_word_corpus.add('<eos>')
    summary_word_corpus.add('<pad>')
    vocalbulary_text += 3
    vocalbulary_summary += 3
    vocalbulary_all += 3
    all_word_to_id = dict(zip(all_word_corpus, range(vocalbulary_all)))
    text_word_to_id = dict(zip(text_word_corpus, range(vocalbulary_text)))
    summary_word_to_id = dict(zip(summary_word_corpus,
                                  range(vocalbulary_summary)))
    all_id_to_word = dict(zip(all_word_to_id.values(), all_word_to_id.keys()))
    text_id_to_word = dict(zip(text_word_to_id.values(),
                               text_word_to_id.keys()))
    summary_id_to_word = dict(zip(summary_word_to_id.values(),
                                  summary_word_to_id.keys()))

    pad = '<pad>'
    # create training set
    for i in trainset_idx:
        text = ['<sos>']
        text += word_tokenize(cleaned_data['text'][i])
        text += ['<eos>']
        for j in range(max_text_length - len(text)):
            text.append(pad)
        text = [text_word_to_id[word] for word in text]

        summary = []
        for j in range(config['window']-1):
            summary.append(pad)
        summary += ['<sos>']
        summary += word_tokenize(cleaned_data['summary'][i])
        summary += ['<eos>']
        try:
            summary = [summary_word_to_id[word] for word in summary]
        except KeyError:
            print(word)
            pdb.set_trace()
        for j in range(config['window'], len(summary)):
            sum_tem = summary[j-config['window']:j]
            label_tem = summary[j]
            with open(os.path.join(exp_data_path,
                                   "dataset",
                                   "{}_text.npy".format(
                                           sample_idx)),
                      "wb") as f:
                pickle.dump(text, f)
            with open(os.path.join(exp_data_path,
                                   "dataset",
                                   "{}_sum.npy".format(
                                           sample_idx)),
                      "wb") as f:
                pickle.dump(sum_tem, f)
            partition['train'].append(sample_idx)
            labels[sample_idx] = label_tem
            sample_idx += 1

    # create validation set
    for i in valiset_idx:
        text = ['<sos>']
        text += word_tokenize(cleaned_data['text'][i])
        text += ['<eos>']
        for j in range(max_text_length - len(text)):
            text.append(pad)
        text = [text_word_to_id[word] for word in text]

        summary = []
        for j in range(config['window']-1):
            summary.append(pad)
        summary += ['<sos>']
        summary += word_tokenize(cleaned_data['summary'][i])
        summary += ['<eos>']
        summary = [summary_word_to_id[word] for word in summary]
        for j in range(config['window'], len(summary)):
            sum_tem = summary[j-config['window']:j]
            label_tem = summary[j]
            with open(os.path.join(exp_data_path,
                                   "dataset",
                                   "{}_text.npy".format(
                                           sample_idx)),
                      "wb") as f:
                pickle.dump(text, f)
            with open(os.path.join(exp_data_path,
                                   "dataset",
                                   "{}_sum.npy".format(
                                           sample_idx)),
                      "wb") as f:
                pickle.dump(sum_tem, f)
            partition['validation'].append(sample_idx)
            labels[sample_idx] = label_tem
            sample_idx += 1

    # create test set
    testset = dict()
    testset['text_actual'] = []  # actual text document
    testset['text'] = []  # will be transformed to input to model
    testset['gt_summary_actual'] = []  # actual summary document
    testset['gt_summary'] = []  # will be compared with model output
    for i in testset_idx:
        testset['text_actual'].append(cleaned_data['text'][i])

        text = ['<sos>']
        text += word_tokenize(testset['text_actual'][-1])
        text += ['<eos>']
        for j in range(max_text_length - len(text)):
            text.append(pad)
        text = [text_word_to_id[word] for word in text]
        testset['text'].append(text)

        testset['gt_summary_actual'].append(cleaned_data['summary'][i])
        testset['gt_summary'].append(word_tokenize(
                testset['gt_summary_actual'][-1]) + ['<eos>'])

    with open(os.path.join(exp_data_path, "testset.pickle"), "wb") as f:
        pickle.dump(testset, f)
    print_and_log(logger, "testset.pickle saved!")
    with open(os.path.join(exp_data_path, 'partition.json'), 'w') as f:
        json.dump(partition, f)
    print_and_log(logger, "partition.json saved!")
    with open(os.path.join(exp_data_path, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    print_and_log(logger, "labels.json saved!")

    args = (partition, labels, testset,
            all_word_corpus, text_word_corpus, summary_word_corpus,
            all_word_to_id, text_word_to_id, summary_word_to_id,
            all_id_to_word, text_id_to_word, summary_id_to_word,
            trainset_idx, valiset_idx, testset_idx,
            vocalbulary_all, vocalbulary_text, vocalbulary_summary,
            max_text_length, max_summary_length)
    return args


if config['create_partition']:
    args = train_vali_test_creation()
    (partition, labels, testset,
        all_word_corpus, text_word_corpus, summary_word_corpus,
        all_word_to_id, text_word_to_id, summary_word_to_id,
        all_id_to_word, text_id_to_word, summary_id_to_word,
        trainset_idx, valiset_idx, testset_idx,
        vocalbulary_all, vocalbulary_text, vocalbulary_summary,
        max_text_length, max_summary_length) = args
else:
    with open(os.path.join(exp_data_path, 'partition.json'), 'r') as f:
        partition = json.load(f)
    print('partition.json loaded!')
    with open(os.path.join(exp_data_path, 'labels.json'), 'r') as f:
        labels = json.load(f)
    print('labels.json loaded!')
    with open(os.path.join(exp_data_path, 'dataset_stat.json'), 'r') as f:
        dataset_stat = json.load(f)
    print_and_log(logger, "dataset_stat.json loaded!")
    vocalbulary_all = dataset_stat['vocalbulary_all']
    vocalbulary_text = dataset_stat['vocalbulary_text']
    vocalbulary_summary = dataset_stat['vocalbulary_summary']
    max_text_length = dataset_stat['max_text_length']
    max_summary_length = dataset_stat['max_summary_length']


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_path, list_IDs, labels,
                 dim, window_size, n_classes, batch_size=256,
                 shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.window_size = window_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataset_path = dataset_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x_text = np.empty((self.batch_size, self.dim))
        x_sum = np.empty((self.batch_size, self.window_size))
        y = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x_text[i, ] = np.load(os.path.join(
                        self.dataset_path, "{}_text".format(ID) + '.npy'))
            x_sum[i, ] = np.load(os.path.join(
                        self.dataset_path, "{}_sum".format(ID) + '.npy'))

            # Store class
            y[i] = self.labels[str(ID)]

        y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        X = [x_text, x_sum]
        return X, y


# LTSM model architecture
# article input model
inputs1 = Input(shape=(max_text_length,))
article1 = Embedding(vocalbulary_text, 256)(inputs1)
article2 = CuDNNLSTM(1024)(article1)
article3 = RepeatVector(config['window'])(article2)
# summary input model
inputs2 = Input(shape=(config['window'],))
summ1 = Embedding(vocalbulary_summary, 256)(inputs2)
summ2 = CuDNNLSTM(1024)(summ1)
summ3 = Dense(1024, activation="relu")(summ2)
summ4 = Dropout(0.8)(summ3)
summ5 = RepeatVector(config['window'])(summ4)
# decoder model
decoder1 = Concatenate()([article3, summ5])
decoder2 = CuDNNLSTM(1024)(decoder1)
decoder3 = Dense(1024, name="dense_two")(decoder2)
decoder4 = Dropout(0.8)(decoder3)
outputs = Dense(vocalbulary_summary, activation='softmax')(decoder4)

# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())
callbacks = [EarlyStopping(monitor='val_loss', patience=100),
             ModelCheckpoint(
                     filepath=exp_data_path + '/model-{epoch:02d}.hdf5',
                     monitor='val_loss',
                     verbose=2)]

if args.run_opt == 1:
    num_epochs = 800
    train_data_generator = DataGenerator(os.path.join(exp_data_path,
                                                      "dataset"),
                                         partition['train'],
                                         labels, batch_size=256,
                                         dim=max_text_length,
                                         window_size=config['window'],
                                         n_classes=vocalbulary_summary,
                                         shuffle=False)
    vali_data_generator = DataGenerator(os.path.join(exp_data_path,
                                                     "dataset"),
                                        partition['validation'],
                                        labels, batch_size=256,
                                        dim=max_text_length,
                                        window_size=config['window'],
                                        n_classes=vocalbulary_summary,
                                        shuffle=False)
    model.fit_generator(generator=train_data_generator,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        validation_data=vali_data_generator,
                        use_multiprocessing=True,
                        workers=8,
                        shuffle=False)
    model.save(exp_data_path + "final_model.hdf5")
#elif args.run_opt == 2:
#    model = load_model(data_path + "\model-40.hdf5")
#    dummy_iters = 40
#    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
#                                                     skip_step=1)
#    print("Training data:")
#    for i in range(dummy_iters):
#        dummy = next(example_training_generator.generate())
#    num_predict = 10
#    true_print_out = "Actual words: "
#    pred_print_out = "Predicted words: "
#    for i in range(num_predict):
#        data = next(example_training_generator.generate())
#        prediction = model.predict(data[0])
#        predict_word = np.argmax(prediction[:, num_steps-1, :])
#        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
#        pred_print_out += reversed_dictionary[predict_word] + " "
#    print(true_print_out)
#    print(pred_print_out)
#    # test data set
#    dummy_iters = 40
#    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
#                                                     skip_step=1)
#    print("Test data:")
#    for i in range(dummy_iters):
#        dummy = next(example_test_generator.generate())
#    num_predict = 10
#    true_print_out = "Actual words: "
#    pred_print_out = "Predicted words: "
#    for i in range(num_predict):
#        data = next(example_test_generator.generate())
#        prediction = model.predict(data[0])
#        predict_word = np.argmax(prediction[:, num_steps - 1, :])
#        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
#        pred_print_out += reversed_dictionary[predict_word] + " "
#    print(true_print_out)
#    print(pred_print_out)
