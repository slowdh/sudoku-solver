import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


# lets train simple neural network model

# first of all, we need data set for sudoku
# make dataset: svhn digits (for 0 ~ 9) + noise images (for None or blank cell)
def load_data(noise_level=3., mode='mnist', add_zero=False):
    if mode == 'svhn':
        train = loadmat('data/svhn_train_32x32')
        test = loadmat('data/svhn_test_32x32')
        x_train, y_train = train['X'], train['y']
        x_test, y_test = test['X'], test['y']
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        y_train[y_train == 10] -= 10
        y_test[y_test == 10] -= 10
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        data = ((x_train, y_train), (x_test, y_test))

        if add_zero is True:
            noise_imgs_train = np.random.uniform(0., noise_level, (7000, 32, 32, 3))
            noise_imgs_test = np.random.uniform(0., noise_level, (2500, 32, 32, 3))
            noise_label_train = np.full((7000,), 10, dtype="uint8")
            noise_label_test = np.full((2500,), 10, dtype="uint8")

            x_train_concatenated = np.concatenate((x_train, noise_imgs_train), axis=0)
            y_train_concatenated = np.concatenate((y_train, noise_label_train), axis=0)
            x_test_concatenated = np.concatenate((x_test, noise_imgs_test), axis=0)
            y_test_concatenated = np.concatenate((y_test, noise_label_test), axis=0)

            shuffled_indices_train = np.random.permutation(x_train_concatenated.shape[0])
            shuffled_indices_test = np.random.permutation(x_test_concatenated.shape[0])
            x_train_shuffled = x_train_concatenated[shuffled_indices_train]
            y_train_shuffled = y_train_concatenated[shuffled_indices_train]
            x_test_shuffled = x_test_concatenated[shuffled_indices_test]
            y_test_shuffled = y_test_concatenated[shuffled_indices_test]
            data = ((x_train_shuffled, y_train_shuffled), (x_test_shuffled, y_test_shuffled))

    elif mode == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        data = ((x_train, y_train), (x_test, y_test))

    return data

# build model!
def get_light_model(input_shape, add_zero=False):
    num_last_units = 10 if add_zero is False else 11

    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling2D(2),
        Conv2D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, 1, padding='same', activation='relu'),
        Flatten(),
        Dense(num_last_units, activation='softmax')
    ])
    return model

def get_heavy_model(input_shape, add_zero=False):
    num_last_units = 10 if add_zero is False else 11

    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.3),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu'),
        Conv2D(32, 1, padding='same', activation='relu'),
        Flatten(),
        Dense(num_last_units, activation='softmax')
    ])
    return model

def get_transfer_model(model_path='weights/weights_mnist.h5'):
    original_model = load_model(model_path)
    original_model.trainable = False

    inputs = original_model.input
    flatten_output = original_model.layers[-2].output
    outputs = Dense(10, activation='softmax')(flatten_output)
    transfer_model = Model(inputs, outputs)
    return transfer_model

def train_model(data, model_path, save_path, lr=3e-4):
    model = load_model(model_path)
    check_point = tf.keras.callbacks.ModelCheckpoint(filepath=save_path)
    (x_train, y_train), (x_test, y_test) = data
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test), callbacks=[check_point])
