import numpy as np


def get_next_possible_space(board):
    for i in range(9):
        for j in range(9):
            if board[i,j] == 0:
                return (i, j)
    return (None, None)

def is_possible(i, j, n):
    global board
    if n in board[i,:] or n in board[:,j]:
        return False
    x0 = (i // 3) * 3
    y0 = (j // 3) * 3
    if n in board[x0:x0+3, y0:y0+3]:
        return False
    return True

def solve(board):
    i, j = get_next_possible_space(board)
    if i is None:
        return (True, board)
    for num in range(1, 10):
        if is_possible(i, j, num):
            board[i,j] = num
            boolean, new_board = solve(board)
            if boolean:
                return (True, new_board)
            board[i, j] = 0
    return (False, board)

def pretty_print(board):
    for idx, row in enumerate(board):
        if idx == 3 or idx == 6:
            print("---------------------")
        for i, num in enumerate(row):
            if i == 3 or i == 6:
                print('|', end=" ")
            print(num, end=" ")
        print()
    print()


sample_board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,0],
    [1,2,0,0,0,7,4,0,0],
    [4,0,0,0,1,0,0,0,7]
]
board = np.array(sample_board)
pretty_print(solve(board)[1])

"""
>>>
    7 8 5 | 4 3 9 | 1 2 6 
    6 1 2 | 8 7 5 | 3 4 9 
    3 4 9 | 6 2 1 | 5 7 8 
    ---------------------
    8 5 7 | 9 4 3 | 2 6 1 
    2 6 1 | 7 5 8 | 9 3 4 
    9 3 4 | 1 6 2 | 7 8 5 
    ---------------------
    5 7 8 | 3 9 4 | 6 1 2 
    1 2 6 | 5 8 7 | 4 9 3 
    4 9 3 | 2 1 6 | 8 5 7 
"""
