import keras
from keras import layers
from keras import regularizers
import tensorflow as tf
import numpy as np
from scipy import stats
import os
import math
from sklearn.preprocessing import minmax_scale
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split


def model_1(d):
    def build_encoder_model(d, input_embb):
        encoded = layers.Dense(int(d/8), activation='relu', activity_regularizer=regularizers.l1(10e-5), name='encode_1')(input_embb)
        return encoded
    
    
    def build_decoder_model(encoded, d):
        decoded = layers.Dense(int(d/8), activation='relu', name='decode_1')(encoded)
        decoded = layers.Dense(d, activation='relu', name='decode_2')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(21,170,), name='input')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    
    shared = layers.Dense(int(d/16), activation='relu', name='shared')(encoded_1)
    
    decoded_1 = build_decoder_model(shared, d)
    
    autoencoder = keras.Model(input_embb_1, decoded_1, name='autoencoder_1')
        
    encoder = keras.Model(input_embb_1, shared, name='encoder_1')
    
    return autoencoder, encoder


def model_2(d):
    def build_encoder_model(d, input_embb):
        encoded = layers.Dense(int(d/8), activation='relu', activity_regularizer=regularizers.l1(10e-5), name='encode_1')(input_embb)
    
        return encoded
    
    
    def build_decoder_model(encoded, d):
        decoded = layers.Dense(int(d/8), activation='relu', name='decode_1')(encoded)
        decoded = layers.Dense(d, activation='relu', name='decode_2')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(21,170,), name='input')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    
    shared = layers.Dense(int(d/16), activation='relu', name='shared')(encoded_1)
    shared_1 = layers.Dense(int(d/170),activation='relu', name='shared_1')(shared)
    
    decoded_1 = build_decoder_model(shared_1, d)
    
    autoencoder = keras.Model(input_embb_1, decoded_1, name='autoencoder_2')
        
    encoder = keras.Model(input_embb_1,
                          shared_1, name='encoder_2')
    
    return autoencoder, encoder


def model_12(d):  #output of model_1 (decoded) is the input for this model
    def build_encoder_model(d, input_embb):
        encoded = layers.Dense(int(d/8), activation='relu', activity_regularizer=regularizers.l1(10e-5), name='encode_1')(input_embb)
    
        return encoded
    
    
    def build_decoder_model(encoded, d):
        decoded = layers.Dense(int(d/8), activation='relu', name='decode_1')(encoded)
        decoded = layers.Dense(d, activation='relu', name='decode_2')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(21,170,), name='input')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    
    shared = layers.Dense(int(d/16), activation='relu', name='shared')(encoded_1)
    shared_1 = layers.Dense(int(d/90), activation='relu', name='shared_1')(shared)
    shared_2 = layers.Dense(int(d/16), activation='relu', name='shared_2')(shared_1)
    
    decoded_1 = build_decoder_model(shared_2, d)
    
    autoencoder = keras.Model(input_embb_1, decoded_1, name='autoencoder_12')
        
    encoder = keras.Model(input_embb_1,
                          shared_1, name='encoder_12')
    
    return autoencoder, encoder


def train_model(model_input, autoencoder, encoder, epoch):
        
    optimizer = keras.optimizers.Adam(lr=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    autoencoder.fit(model_input, model_input,
                epochs=epoch,
                batch_size=16,
                shuffle=True,
                   )
    
    denoised_signal = autoencoder.predict(model_input)

    return autoencoder, encoder, denoised_signal


def gen_features(raw_data, encoder):
    features = encoder.predict(raw_data)
    features = features.reshape((features.shape[0],int(features.shape[1]*features.shape[2])))
    
    return features


def gen_dataset(features, c_1, c_2, c_3):
    l1 = c_1
    l2 = l1 + c_2
    dataset = pd.DataFrame(features)
    dataset['label'] = 0
    dataset.loc[:l1, 'label'] = 1
    dataset.loc[l1:l2, 'label'] = 2
    dataset.loc[l2:, 'label'] = 3
    
    return dataset


def get_train_test(dataset):
    train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=0)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    X_train = train[train.columns[:-1]]
    y_train = train['label']

    X_test = test[test.columns[:-1]]
    y_test = test['label']
    
    return X_train, y_train, X_test, y_test
