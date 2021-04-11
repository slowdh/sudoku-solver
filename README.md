# sudoku-solver

## How it works?
 <img src="https://github.com/slowdh/sudoku-solver/blob/main/data/images/explain.jpg">

Now, board detection part is done with open-cv library. It detects square shape contour with largest area on image, and flatten out perspective with warpPerspective function.

Digit classification part is done with deep learning. Model is pre-trained with svhn dataset and done transfer learning with custom dataset.

Sudoku solving part is done with simple backtracking algorithm.


## Examples

```python
import cv2
from tensorflow.keras.models import load_model
from ocr import solve_sudoku, solve_sudoku_on_camera

# load image and digit classifier model
test_img_path = 'data/images/test-images/1.jpg'
test_model_path = 'weights/model.h5'
test_img = cv2.imread(test_img_path)
test_model = load_model(test_model_path)

# if want to test on single image
solve_sudoku(image=test_img, ocr_model=test_model, show_output=True)
# want to test on webcam or video input? then try,
solve_sudoku_on_camera(test_model, video_source=0, size=None, save_path=None)

```

## Test cases
<img src="https://github.com/slowdh/sudoku-solver/blob/main/data/images/testcase.jpg">


## TODO

* Update board detection part with deeplearning -> for even more robust model!
* Try end-to-end multitask model for sharing computation between betecting board / classification of each digit.
* Try get more data with current model.
