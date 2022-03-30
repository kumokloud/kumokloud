import numpy as np
import math
import matplotlib.pyplot as plt

# 初期値
J = np.array([[114, 0, 0],
              [0, 86, 0],
              [0, 0, 87]])
p1 = -0.1
p2 = 0.5
p3 = 1.0

# 目標のMRP
pd1 = 0
pd2 = 0
pd3 = 0
pd = np.array([[pd1], [pd2], [pd3]])

n = np.array([[1], [1], [1]])
theta1 = math.atan(p1)*4*(180/math.pi)
print(theta1)
