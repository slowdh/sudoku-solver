import numpy as np


class Sudoku:
    def __init__(self, board):
        self.board = np.array(board)

    def __str__(self):
        ret = []
        for idx, row in enumerate(self.board):
            pretty_row = ''
            if idx == 3 or idx == 6:
                ret.append("---------------------")
            for i, num in enumerate(row):
                if i == 3 or i == 6:
                    pretty_row += '| '
                pretty_row += str(num) + ' '
            ret.append(pretty_row)
        return '\n'.join(ret)

    def is_valid_board(self):
        def _is_dup_in(arr):
            for i in range(1, 10):
                if len(np.where(arr == i)[0]) > 1:
                    return True
            return False

        # check dup in row, column
        for i in range(9):
            if _is_dup_in(self.board[i, :]):
                return False
            if _is_dup_in(self.board[:, i]):
                return False
        # check dup in boxes
        for i in range(3):
            for j in range(3):
                if _is_dup_in(self.board[3*i: 3*i+3, 3*j: 3*j+3]):
                    return False
        return True

    def is_valid_cell(self, i, j, n):
        if n in self.board[i, :] or n in self.board[:, j]:
            return False
        x = (i // 3) * 3
        y = (j // 3) * 3
        if n in self.board[x:x + 3, y:y + 3]:
            return False
        return True

    def solve_helper(self):
        for i in range(9):
            for j in range(9):
                if self.board[i, j] == 0:
                    for n in range(1, 10):
                        if self.is_valid_cell(i, j, n):
                            self.board[i, j] = n
                            if self.solve_helper():
                                return True
                            self.board[i, j] = 0
                    return False
        return True

    def solve(self):
        if not self.is_valid_board():
            boolean = False
        else:
            if self.solve_helper():
                boolean = True
            else:
                boolean = False
        return boolean, self.board
