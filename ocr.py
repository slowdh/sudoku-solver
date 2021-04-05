import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
from sudoku import Sudoku


def get_distance(pt1, pt2):
    sub = pt2 - pt1
    distance = np.sqrt(np.sum(sub ** 2))
    return distance

def order_points(points):
    ret = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    d = np.diff(points, axis=1)
    ret[0] = points[np.argmin(s)] # top left
    ret[1] = points[np.argmin(d)] # top right
    ret[2] = points[np.argmax(s)] # bottom right
    ret[3] = points[np.argmax(d)] # bottom left
    return ret

def get_destination_points_and_size(points):
    horizontal_top = get_distance(points[0], points[1])
    horizontal_bottom = get_distance(points[2], points[3])
    vertical_left = get_distance(points[0], points[3])
    vertical_right = get_distance(points[1], points[2])
    max_horizontal = int(max(horizontal_top, horizontal_bottom))
    max_vertical = int(max(vertical_left, vertical_right))

    destination = np.array([
        [0, 0],
        [max_horizontal, 0],
        [max_horizontal, max_vertical],
        [0, max_vertical]], dtype="float32")
    return destination, (max_horizontal, max_vertical)

def four_point_perspective_transform(image, points):
    points = order_points(points)
    destination, size = get_destination_points_and_size(points)
    mat = cv2.getPerspectiveTransform(points, destination)
    transformed = cv2.warpPerspective(image, mat, size)
    return transformed

def find_board_contour(image):
    # preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # to binary.
    # change parameter C if image condition is different
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=3)
    thresh = cv2.bitwise_not(thresh)

    # get_contour
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # approximation of complex shape contours -> find rectangle with largest area!
    epsilon = 0.02
    puzzle_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon * perimeter, True)
        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        print("Could not find proper puzzle contour")
    return order_points(puzzle_contour.reshape((4, 2)))

def get_warped_board(image, contour):
    contour = find_board_contour(image)
    warped_board = four_point_perspective_transform(image, contour)
    return warped_board

def extract_digit_from_cell(cell):
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    thresh = clear_border(thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.03:
        return None
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    return digit

def ocr_board(warped, model, input_shape, debugging=False):
    img = cv2.resize(warped, dsize=(input_shape[1] * 9, input_shape[1] * 9), interpolation=cv2.INTER_LINEAR)
    board = np.zeros(shape=(9, 9), dtype='uint8')
    mapping_dict = {i:j for i, j in zip(range(11), list(range(10)) + [0])}
    step = input_shape[0]
    for i in range(9):
        for j in range(9):
            x_start = i * step
            x_end = (i + 1) * step
            y_start = j * step
            y_end = (j + 1) * step

            img_input = img[x_start:x_end, y_start:y_end]
            digit = extract_digit_from_cell(img_input)
            if digit is None:
                digit_ocr = 0
            else:
                img_input = np.expand_dims(digit, axis=2)
                if input_shape[-1] == 3:
                    img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
                img_tensor = np.expand_dims(img_input, axis=0)
                argmax = np.argmax(model(img_tensor))
                digit_ocr = mapping_dict[argmax]
                if debugging:
                    cv2.imshow("window", img_input)
                    print(digit_ocr)
                    cv2.waitKey(0)
            board[i, j] = digit_ocr
    return board

def visualize_output_on_warped_image(original, solved, warped):
    h, w = warped.shape[0] // 9, warped.shape[1] // 9
    for i in range(9):
        for j in range(9):
            if original[i, j] == 0:
                x_start = j * w
                y_start = i * h
                x_diff = int(w * 0.4)
                y_diff = int(h * 0.7)
                x_org = x_start + x_diff
                y_org = y_start + y_diff
                cv2.putText(warped, str(solved[i, j]), (x_org, y_org), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    return warped

def get_points(img):
    h, w = img.shape[:2]
    points = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]], dtype="float32")
    return points

def solve_sudoku(image_path='test-images/opencv_sudoku_puzzle_sudoku_puzzle.jpg', ocr_model_path='weights/transferred.h5'):
    model = load_model(ocr_model_path)
    img = cv2.imread(image_path)

    puzzle_countour = find_board_contour(img)
    destination, size = get_destination_points_and_size(puzzle_countour)
    warped = get_warped_board(img, destination)
    board = ocr_board(warped, model, (28, 28, 1), debugging=False)
    sudoku = Sudoku(board.copy())
    sudoku.solve()
    solved = sudoku.board
    warped = visualize_output_on_warped_image(board, solved, warped)
    warped_to_original = four_point_perspective_transform(warped, puzzle_countour)
    
    # WIP #
    points = get_points(warped)
    mat = cv2.getPerspectiveTransform(points, puzzle_countour)
    transformed = cv2.warpPerspective(warped, mat, img.shape[:2])

    cv2.imshow("Answer", overlayed)
    cv2.waitKey(0)


# ### test section
# model = load_model('weights/transferred.h5')
# img = cv2.imread('test-images/opencv_sudoku_puzzle_sudoku_puzzle.jpg')
#
# puzzle_countour = find_board_contour(img)
# original = order_points(puzzle_countour)
# destination, size = get_destination_points_and_size(puzzle_countour)
# warped = get_warped_board(img, destination)
# board = ocr_board(warped, model, (28, 28, 1), debugging=False)
# sudoku = Sudoku(board.copy())
# sudoku.solve()
# solved = sudoku.board
# warped = visualize_output_on_image(board, solved, warped)
#
# print(board)
# cv2.imshow("a", warped)
# cv2.waitKey(0)

solve_sudoku()
