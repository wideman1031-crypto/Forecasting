import math, datetime, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, LSTM, LayerNormalization, MultiHeadAttention
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


class Helformer:
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, args):
        self.units = 64
        self.num_blocks = 4
        self.num_heads = 4
        self.head_size = 16
        self.dropout_rate = 0.1
        self.epochs = 1000
        self.learning_rate = 0.0001
        self.loss = 'mean_squared_error'
        self.batch_size = 32
        self.is_model_created = False
        self.model = None

    def mish(self, x):
        return x * K.tanh(K.softplus(x))

    # # 8. Helformer Model Definition
    def create_model(self, input_shape, units=32, num_blocks=2, num_heads=4, head_size=32, dropout_rate=0.1):
        inputs = Input(shape=(1, input_shape))
        x = inputs
        for _ in range(self.num_blocks):
            x_norm1 = LayerNormalization(epsilon=1e-6)(x)
            attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout_rate)(x_norm1, x_norm1)
            x = x + attn_output
            x_norm2 = LayerNormalization(epsilon=1e-6)(x)
            x = x + x_norm2
        x = LSTM(self.units, activation= self.mish, return_sequences=False)(x)
        outputs = Dense(1)(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.is_model_created = True
    
    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]

        if self.is_model_created == False:
            self.create_model(train_x.shape[1])

        train_x = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint('helformer_best.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        self.model.fit(train_x, train_y,
        validation_split=0.2,
        epochs=self.epochs,
        batch_size=self.batch_size,
        verbose=0,
        callbacks=[reduce_lr, checkpoint])

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        test_x = self.sc_in.transform(test_x)
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        pred_y = self.model.predict(test_x)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y.reshape(-1)