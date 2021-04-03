import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


# lets train simple neural network model

# first of all, we need data set for sudoku
# svhn digits (for 0 ~ 9) + noise images (for None or blank cell) should work.
def load_data(noise_level=3.):
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
    return (x_train_shuffled, y_train_shuffled), (x_test_shuffled, y_test_shuffled)

# lets build model!
# total params: 61,963
def get_light_model(lr = 3e-4):
    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
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
        Dense(11, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# total_params: 314,571
def get_heavy_model(lr = 3e-4):
    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2),
        Conv2D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2),
        Dropout(0.3),
        Conv2D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(32, 1, padding='same', activation='relu'),
        Flatten(),
        Dense(11, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# train time!
def train_model(file_path='weights/weights.h5', lr = 3e-5):
    check_point = tf.keras.callbacks.ModelCheckpoint(filepath=file_path)
    (x_train, y_train), (x_test, y_test) = load_data()
    # model = get_light_model()
    model = tf.keras.models.load_model(filepath=file_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=15, verbose=1, validation_data=(x_test, y_test), callbacks=[check_point])


# train_model()

# want to test on empty cell?
# model = tf.keras.models.load_model('weights/weights_light_v1.h5')
# none_img = np.full((1, 32, 32, 3), 0, dtype='uint8')
# prediction = np.argmax(model.predict(none_img))
# print(prediction)
# # -> 10!
