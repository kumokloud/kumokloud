import numpy as np
import cvxpy as cp

# Define the Data.
A = np.array([[3, 2], [2, 6]])
b = np.array([5, 8])
c = np.array([2, 3])

# Define and solve the CVXPY problem.
x = cp.Variable(2)  # 決定変数の次元
objective = cp.Minimize(c.T@x)  # 目的関数
constraints = [A@x >= b, x >= 0]  # 制約条件
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)  # 最適値
print("A solution x is")  # 最適解
print(x.value)
print("A dual solution is")  # 双対問題の最適解
# prob.constraints[制約条件の何式目を使用するか].dual_value:制約条件付きでの最適解
print(prob.constraints[0].dual_value)
