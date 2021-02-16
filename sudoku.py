import numpy as np


class Sudoku:
    def __init__(self, board):
        self.board = board

    def __str__(self):
        ret = []
        for idx, row in enumerate(board):
            pretty_row = ''
            if idx == 3 or idx == 6:
                ret.append("---------------------")
            for i, num in enumerate(row):
                if i == 3 or i == 6:
                    pretty_row += '| '
                pretty_row += str(num) + ' '
            ret.append(pretty_row)
        return '\n'.join(ret)

    def get_next_valid_space(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return (i, j)
        return (None, None)

    def is_valid(self, i, j, n):
        if n in self.board[i, :] or n in self.board[:, j]:
            return False
        x = (i // 3) * 3
        y = (j // 3) * 3
        if n in board[x:x + 3, y:y + 3]:
            return False
        return True

    def solve(self):
        i, j = self.get_next_valid_space()
        if i is not None:
            for n in range(1, 10):
                if self.is_valid(i, j, n):
                    self.board[i][j] = n
                    self.solve()
                    self.board[i][j] = 0
            return
        print(self)


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
a = Sudoku(board)
a.solve()

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
