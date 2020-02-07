import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_regressor(neurons=1, activation='linear'):
    model = Sequential()
    model.add(Dense(units=neurons, activation=activation, input_dim=3))
    model.add(Dense(units=3, activation='linear'))

    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

    model.compile(loss='mean_squared_error',
            optimizer=sgd,
            metrics=[r2_score, "mae"])

    return model

def create_classifier(learn_rate=0.1, clipvalue=0.9, neurons=500):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", input_dim=3))
    model.add(Dense(units=4, activation="softmax"))

    # Stochastic gradient descent optimizer: learning rate, clip value
    sgd = optimizers.SGD(lr=learn_rate, clipvalue=clipvalue)

    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    return model
