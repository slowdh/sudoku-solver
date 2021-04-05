import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense

# generating .npy file from raw dataset folder
def generate(file_path = 'data/assets/'):
    data = None
    target = None
    for i in range(10):
        paths = glob.glob(file_path + str(i) + '/*')
        l = len(paths)
        target_curr = np.array([i] * l)
        if target is None:
            target = target_curr
        else:
            target = np.concatenate((target, target_curr), axis=0)

        for path in paths:
            img = np.array(Image.open(path).convert('L'))[np.newaxis, ..., np.newaxis]
            assert img.shape == (1, 28, 28, 1)
            if data is None:
                data = img
            else:
                data = np.concatenate((data, img), axis=0)

    np.save(file_path + 'data.npy', data)
    np.save(file_path + 'target.npy', target)

def get_trainable_dataset(file_path = 'data/assets/', test_split=0.1):
    x, y = np.load(file_path + 'data.npy'), np.load(file_path + 'target.npy')
    permutation = np.random.permutation(len(x))
    x, y = x[permutation], y[permutation]

    split_idx = int(len(x) * (1 - test_split))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return (x_train, x_test), (y_train, y_test)

def get_transfer_model(model_path='weights/weights_mnist.h5'):
    original_model = load_model(model_path)
    original_model.trainable = False

    inputs = original_model.input
    flatten_output = original_model.layers[-2].output
    outputs = Dense(10, activation='softmax')(flatten_output)
    transfer_model = Model(inputs, outputs)
    return transfer_model

(x_train, x_test), (y_train, y_test) = get_trainable_dataset()
# transfer_model = get_transfer_model()
# transfer_model.summary()
transfer_model = load_model('weights/transferred.h5')
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(3e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint('weights/transferred.h5', save_best_only=True)
transfer_model.fit(x=x_train, y=y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test), callbacks=[checkpoint])