import glob
import cv2
import numpy as np
from PIL import Image
from ocr import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


# first, detect board contour and split cell
# put image in folder name {argmax} of prediction with image name as prediction probability
# this would make easier to detect misclassified images
def split_cell_img_and_classify(warped, model, save_path=None, check_cell=False, skip_zero=False):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dsize=(252, 252), interpolation=cv2.INTER_LINEAR)
    step = gray.shape[0] // 9

    for i in range(9):
        for j in range(9):
            x_start = i * step
            x_end = (i + 1) * step
            y_start = j * step
            y_end = (j + 1) * step

            cell = np.expand_dims(gray[x_start:x_end, y_start:y_end], axis=2)
            digit = extract_digit_from_cell(cell=cell)
            max_prob = np.random.randn()
            if digit is None:
                digit = 0
                max_prob = np.random.randn(1)[0]

                if check_cell and not skip_zero:
                    print(f"Predicted: {digit}")
                    cv2.imshow("Prediction: " + str(digit), cell)
                    key = cv2.waitKey(0)
                    if key == 13:  # enter key
                        print("    Continue processing.")
                    else:
                        digit = chr(key)
                        print(f"    Changed label to {digit}")
                    cv2.destroyAllWindows()
            else:
                img_input = np.expand_dims(digit, axis=2)
                img_tensor = np.expand_dims(img_input, axis=0)
                prob = model(img_tensor)
                digit = np.argmax(prob)

                if check_cell:
                    concat = np.concatenate((cell, img_input), axis=1)
                    cv2.imshow("Prediction: " + str(digit), concat)
                    print(f"Predicted: {digit}")
                    key = cv2.waitKey(0)
                    if key == 13:  # enter key
                        print("    Continuing processing.")
                    else:
                        digit = chr(key)
                        print(f"    Changed label to {digit}")
                    cv2.destroyAllWindows()

                if save_path is not None:
                    max_prob = K.get_value(prob[0, digit])
                    masked_file_path = save_path + '/masked/' + str(digit) + '/' + str(max_prob)[2:10] + '.jpg'
                    cv2.imwrite(masked_file_path, img_input)

            if save_path is not None:
                original_file_path = save_path + '/original/' + str(digit) + '/' + str(max_prob)[2:10] + '.jpg'
                cv2.imwrite(original_file_path, cell)

def generate_digit_dataset_from_image(image_paths, model_path, save_path=None, check_warped=True, check_cell=False, skip_zero=True):
    model = load_model(model_path)
    for path in image_paths:
        image = cv2.imread(path)

        warped, contour = get_warped_board_with_contour(image=image)
        if contour is not None and check_warped:
            cv2.imshow("Warped output", warped)
            key = cv2.waitKey(0)
            if key == 13: #enter key
                print("Classifying cell...")
                split_cell_img_and_classify(warped, model, save_path, check_cell, skip_zero)
            else:
                print("Skipping current image.")
            cv2.destroyAllWindows()

def print_data_distribution(img_path):
    num_imgs = []
    for i in range(10):
        num = len(glob.glob(img_path + '/' + str(i) + '/*'))
        num_imgs.append(num)

    total = sum(num_imgs)
    print(f"Total: {total} images" + '\n')
    for i in range(10):
        print(f"Label: {i}  num: {num_imgs[i]}  ratio: {int(num_imgs[i] / total * 100)}%")

# generating .npz file from raw dataset folder
def get_np_array_from_imgs(file_path, save_path, seperate_by_label=True, limit=400):
    data = None
    target = None

    for i in range(10):
        paths = glob.glob(file_path + '/' + str(i) + '/*')
        target_curr = np.array([i] * min(len(paths), limit))
        data_curr = None
        for path in paths:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            img = img[np.newaxis, ..., np.newaxis]
            assert img.shape == (1, 28, 28, 1)
            if data_curr is None:
                data_curr = img
            else:
                data_curr = np.concatenate((data_curr, img), axis=0)
        data_curr = data_curr[:limit]

        if seperate_by_label:
            np.savez(save_path + '/' + str(i) + '_dataset.npz', x=data_curr, y=target_curr)
            print("Data saved at:", save_path + '/' + str(i) + '_data.npz')
        else:
            if data is None:
                data = data_curr
            else:
                data = np.concatenate((data, data_curr), axis=0)
            if target is None:
                target = target_curr
            else:
                target = np.concatenate((target, target_curr), axis=0)
    if not seperate_by_label:
        np.savez(save_path + '/dataset.npz', x=data, y=target)
        print("Data saved at:", save_path + '/' + 'data.npz')

def shuffle_and_split(x, y, test_split=0.3):
    permutation = np.random.permutation(len(x))
    x, y = x[permutation], y[permutation]

    split_idx = int(len(x) * (1 - test_split))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return (x_train, y_train), (x_test, y_test)

# there should be a better way than this, but it works in reasonable time.
def modify_dataset_image_shape(imgs, output_size=(28,28)):
    ret = None
    m = imgs.shape[0]
    for i in range(m):
        extracted = imgs[i]
        modified = cv2.resize(extracted, output_size, interpolation=cv2.INTER_LINEAR)[np.newaxis,...]
        if ret is None:
            ret = modified
        else:
            ret = np.concatenate((ret, modified), axis=0)
        print(f"Processed {i + 1} / {m}, Output shape: {ret.shape}")
    return ret
