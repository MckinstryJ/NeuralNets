from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#######################################################################
#######################################################################
#                           STOCK DATA                                #
#######################################################################
#######################################################################
data = pd.read_csv("../../GOOG.csv", header=0)
org_close = data.iloc[:, 5].values
data = data.iloc[:, 2:6].pct_change()[1:]
data += .5

#######################################################################
#######################################################################
#                          Model Creation                             #
#######################################################################
#######################################################################
input_ = Input(shape=(4,))
encoding_dim = 2
encoded = Dense(encoding_dim, activation='relu')(input_)
decoded = Dense(4, activation='sigmoid')(encoded)
encoder = Model(input_, encoded)

autoencoder = Model(input_, decoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#######################################################################
#######################################################################
#                      Training / Testing                             #
#######################################################################
#######################################################################
eighty_ = int(len(data) * .8)
x_train, x_test = data.iloc[:eighty_, :].values, data.iloc[eighty_:, :].values

x_train = x_train.reshape((len(x_train), int(np.prod(x_train.shape[1:]))))
x_test = x_test.reshape((len(x_test), int(np.prod(x_test.shape[1:]))))

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_ = encoder.predict(x_test)
decoded_ = decoder.predict(encoded_)

new_close = [org_close[eighty_]]
for x in range(len(decoded_)):
    new_close.append(new_close[-1] * (decoded_[x][3] - decoded_[0][3] + 1))

#######################################################################
#######################################################################
#                     Visualizing ENCODE / DECODE                     #
#######################################################################
#######################################################################
plt.plot(org_close[eighty_:])
plt.plot(new_close)
plt.show()
