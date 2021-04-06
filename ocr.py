import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
from sudoku import Sudoku


def get_points(img):
    h, w = img.shape[:2]
    points = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]], dtype="float32")
    return points

# give order to unsorted points
def order_points(points):
    ret = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    d = np.diff(points, axis=1)
    ret[0] = points[np.argmin(s)] # top left
    ret[1] = points[np.argmin(d)] # top right
    ret[2] = points[np.argmax(s)] # bottom right
    ret[3] = points[np.argmax(d)] # bottom left
    return ret

def get_distance(pt1, pt2):
    sub = pt2 - pt1
    distance = np.sqrt(np.sum(sub ** 2))
    return int(distance)

def get_adaptive_font_size(cell_height):
    scale = cell_height / 44
    size = cv2.getTextSize('0', cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    return scale, size

def get_transformation_destination_points(cell_size=28):
    destination = np.array([
        [0, 0],
        [cell_size * 9, 0],
        [cell_size * 9, cell_size * 9],
        [0, cell_size * 9]], dtype="float32")
    return destination

def get_transformation_destination_points_and_size(contour):
    horizontal_top = get_distance(contour[0], contour[1])
    horizontal_bottom = get_distance(contour[2], contour[3])
    vertical_left = get_distance(contour[0], contour[3])
    vertical_right = get_distance(contour[1], contour[2])
    max_horizontal = max(horizontal_top, horizontal_bottom)
    max_vertical = max(vertical_left, vertical_right)

    size = (max_horizontal, max_vertical)
    destination = np.array([
        [0, 0],
        [max_horizontal, 0],
        [max_horizontal, max_vertical],
        [0, max_vertical]], dtype="float32")
    return destination, size

def four_point_perspective_transform(image, source, destination, size):
    mat = cv2.getPerspectiveTransform(source, destination)
    transformed = cv2.warpPerspective(image, mat, size)
    return transformed

def add_imgs(source, overlayed):
    gray = cv2.cvtColor(overlayed, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    source_masked = cv2.bitwise_and(source, source, mask=mask_inv)
    output = cv2.add(source_masked, overlayed)
    return output

def extract_digit_from_cell(cell, fill_threshold=0.03):
    # clear boarder and extract contour
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    thresh = clear_border(thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # take largest contour and draw on empty mask
    # if percent_filled is too low, return None
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    percent_filled = cv2.countNonZero(mask) / float(thresh.shape[0] * thresh.shape[1])

    if percent_filled < fill_threshold:
        return None
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    return digit

def get_board_contour(image, c=3, epsilon=0.02):
    # preprocessing. blurr is for smoothing noise on image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # convert to binary.
    # change parameter C if image condition is different
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=c)
    thresh = cv2.bitwise_not(thresh)

    # get_contour
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # smoothing complex shape contours to simple geometry -> find rectangle with largest area
    puzzle_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        print("Could not find proper puzzle contour")
    points_ordered = order_points(puzzle_contour.reshape((4, 2)))
    return points_ordered

def get_warped_board_with_contour(image):
    contour = get_board_contour(image)
    destination, size = get_transformation_destination_points_and_size(contour)
    warped_board = four_point_perspective_transform(image=image, source=contour, destination=destination, size=size)
    return warped_board, contour

def ocr_board(warped, model, debug=False):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dsize=(252, 252), interpolation=cv2.INTER_LINEAR)
    board = np.zeros(shape=(9, 9), dtype='uint8')
    step = gray.shape[0] // 9

    for i in range(9):
        for j in range(9):
            x_start = i * step
            x_end = (i + 1) * step
            y_start = j * step
            y_end = (j + 1) * step

            cell = gray[x_start:x_end, y_start:y_end]
            digit = extract_digit_from_cell(cell=cell)
            if digit is not None:
                img_input = np.expand_dims(digit, axis=2)
                img_tensor = np.expand_dims(img_input, axis=0)
                digit = np.argmax(model(img_tensor))
                board[i, j] = digit
                if debug is True:
                    cv2.imshow("test", img_input)
                    print(digit)
                    cv2.waitKey(0)
    return board

def solve_board(original_board):
    sudoku = Sudoku(original_board.copy())
    boolean = sudoku.solve()
    return boolean, sudoku.board

def visualize_output_on_original_image(original_board, solved, warped, contour, original_img):
    # put solved digit text on blank image where original sudoku board's cell value == 0
    h, w = warped.shape[0] // 9, warped.shape[1] // 9
    font_scale, font_size = get_adaptive_font_size(h)
    zero_indices = zip(*np.where(original_board == 0))
    blank_img = np.zeros(warped.shape, dtype='uint8')
    for i, j in zero_indices:
        x_cen = j * w + w // 2
        y_cen = i * h + h // 2
        x_diff = font_size[0] // 2
        y_diff = font_size[1] // 2
        x_org = x_cen - x_diff
        y_org = y_cen + y_diff
        cv2.putText(blank_img, str(solved[i, j]), (x_org, y_org), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)

    warped_points = get_points(warped)
    digit_transfomed = four_point_perspective_transform(blank_img, warped_points, contour, (original_img.shape[1], original_img.shape[0]))
    overlayed = add_imgs(original_img, digit_transfomed)
    return overlayed

def solve_sudoku(image_path, ocr_model_path, debug=False):
    model = load_model(ocr_model_path)
    img = cv2.imread(image_path)

    warped, contour = get_warped_board_with_contour(image=img)
    unsolved_board = ocr_board(warped, model, debug=debug)
    is_solvable, solved_board = solve_board(unsolved_board)
    if is_solvable is False:
        print("Unable to solve the puzzle. Check OCR")
        print(solved_board)
        img_with_digit = img
    else:
        img_with_digit = visualize_output_on_original_image(unsolved_board, solved_board, warped, contour, img)

    cv2.imshow("Answer", img_with_digit)
    cv2.waitKey(0)


# test time!
solve_sudoku(image_path='test-images/opencv_sudoku_puzzle_sudoku_puzzle.jpg',ocr_model_path='weights/transferred.h5')
