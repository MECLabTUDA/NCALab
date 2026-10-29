import numpy as np

op_sobel_x: np.ndarray = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
op_sobel_y: np.ndarray = op_sobel_x.T
op_laplace: np.ndarray = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])
op_prewitt_x: np.ndarray = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 6
op_prewitt_y: np.ndarray = op_prewitt_x.T
