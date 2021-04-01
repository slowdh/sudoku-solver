import cv2
import numpy as np


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

def get_transformed_points(points):
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
    destination, size = get_transformed_points(points)
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
        raise ValueError("Could not find proper puzzle contour")
    return puzzle_contour.reshape((4, 2))


### test section
img = cv2.imread('test-images/opencv_sudoku_puzzle_sudoku_puzzle.jpg')
puzzle_countour = locate_board(img)
original = order_points(puzzle_countour)
destination, size = get_transformed_points(puzzle_countour)

transformed = four_point_perspective_transform(img, puzzle_countour)
cv2.imshow("aa", img)
cv2.imshow("a", transformed)
cv2.waitKey(0)