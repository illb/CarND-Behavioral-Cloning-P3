from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.regularizers import l2
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), name='normalization'))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), name='crop'))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu', W_regularizer=l2(0.001), name='conv_01'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu', W_regularizer=l2(0.001), name='conv_02'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu', W_regularizer=l2(0.001), name='conv_03'))
model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001), name='conv_04'))
model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001), name='conv_05'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, W_regularizer=l2(0.001), name='fc_01'))
model.add(Dropout(0.7))
model.add(Dense(50, W_regularizer=l2(0.001), name='fc_02'))
model.add(Dropout(0.7))
model.add(Dense(10, W_regularizer=l2(0.001), name='fc_03'))
model.add(Dense(1, name='fc_04'))

name = "04_nvidia_dropout2"

callback_tb = TensorBoard(log_dir='./logs', write_images=True)
callback_cp = ModelCheckpoint(filepath='./checkpoint/train_' + name + '-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1,
                              save_best_only=True, mode='max')
callback_logger = CSVLogger('./result/train_log_' + name + '.csv')
callbacks = [callback_tb, callback_cp, callback_logger]

import data
train_generator, train_batch_len, validation_generator, validation_batch_len, test_generator, test_batch_len = data.generators()

optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
history_object = model.fit_generator(train_generator, samples_per_epoch=train_batch_len,
                                     validation_data=validation_generator, nb_val_samples=validation_batch_len,
                                     verbose=1, nb_epoch=5, callbacks=callbacks)

model.save('model.h5')

print("test loss: {}".format(model.evaluate_generator(test_generator, test_batch_len)))

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./result/train_result_{}.png".format(name))
