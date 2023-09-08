import streamlit as st
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, ReLU
from keras.optimizers import Adam
import keras.backend as K

SCALE = 0.8
HEIGHT = round(560 * SCALE)
WIDTH = round(950 * SCALE)
N_CATEGORIES = 39

SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002,
    decay_steps=10000,
    decay_rate=0.90)

OPTIMIZER = Adam(learning_rate=SCHEDULE)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def load_model():
        
    with tf.device("/GPU:0"):
        # Load the original model
        original_model = tf.keras.models.load_model('model.h5')

        original_model.save_weights('model_weights.h5')

        del original_model

        model = Sequential()

        model.add(Input(shape=(HEIGHT, WIDTH, 1)))

        model.add(Conv2D(filters=16, kernel_size=7, padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ReLU())
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=32, kernel_size=5, padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ReLU())
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=64, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal'))
        model.add(ReLU())
        model.add(MaxPooling2D(pool_size=2))

        model.add(Flatten())

        for neurons in [128]:
            model.add(Dense(neurons, activation='relu', kernel_initializer='he_normal'))
            model.add(Dropout(0.15))

        # Output layer
        model.add(Dense(N_CATEGORIES, activation='softmax', kernel_initializer='he_normal'))

        # Load the saved weights into the new model
        model.load_weights('model_weights.h5')

        model.compile(optimizer = OPTIMIZER, loss = 'categorical_crossentropy')

        return model

def modelRoute():
    
    model = load_model()

    st.write(model.summary())