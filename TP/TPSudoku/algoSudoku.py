import numpy as np

class AlgoSudoku:

    def __init__(self, grid):
        self.__original_grid = np.asarray(grid).copy()
        self.__result_grid = np.asarray(grid).copy()

    def solve(self):
        find = self.find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if self.is_valid(i, (row, col)):
                self.__result_grid[row][col] = i

                if self.solve():
                    return True

                self.__result_grid[row][col] = 0

        return False

    def find_empty(self):
        for i in range(9):
            for j in range(9):
                if self.__result_grid[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(self, num, pos):
        # Check row
        for i in range(9):
            if self.__result_grid[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(9):
            if self.__result_grid[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self.__result_grid[i][j] == num and (i, j) != pos:
                    return False

        return True

    @property
    def grid(self):
        return self.__result_grid

    @property
    def original_grid(self):
        return self.__original_grid