import numpy as np

x = np.array([i for i in range(16)])
x = np.reshape(x, (2, 2, 1, 4))
b = np.array([1, 1, 1, 1])

print(x + b)
