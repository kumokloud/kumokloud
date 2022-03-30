import numpy as np
import cvxpy as cp

# Define the Data.
n = 3  # 決定変数である行列Xのサイズ
p = 2  # 制約条件の個数

C = np.array([[1, 2, 3],
              [2, 9, 0],
              [3, 0, 7]])
A = []
A.append(np.array([[1, 0, 1], [0, 3, 7], [1, 7, 5]]))
A.append(np.array([[0, 2, 8], [2, 6, 0], [8, 0, 4]]))
print(A)

b = []  # bは適当な定数を与えた
b.append(11)
b.append(19)

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n, n), symmetric=True)  # symmetric:対称性
# Define the objective and constraints
objective = cp.Minimize(cp.trace(C@X))
constraints = []
constraints = [X >> 0]  # The operator >> denotes matrix inequality.
constraints += [cp.trace(A[i]@X) == b[i] for i in range(p)]  # trace:行列の対角成分
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)
