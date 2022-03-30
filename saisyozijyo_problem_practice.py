import numpy as np
import cvxpy as cp

# Define the data.
C = np.array([[0.9501, 0.7620, 0.6153, 0.4057],
              [0.2311, 0.4564, 0.7919, 0.9354],
              [0.6068, 0.0185, 0.9218, 0.9169],
              [0.4859, 0.8214, 0.7382, 0.4102],
              [0.8912, 0.4447, 0.1762, 0.8936]])
d = np.array([0.0578, 0.3528, 0.8131, 0.0098, 0.1388])

A = np.array([[0.2027, 0.2721, 0.7467, 0.4659],
              [0.1987, 0.1988, 0.4450, 0.4186],
              [0.6037, 0.0152, 0.9318, 0.8462]])
b = np.array([0.5251, 0.2026, 0.6721])

# Define and solve the CVXPY problem.
x = cp.Variable(4)  # 決定変数の次元
objective = cp.sum_squares(C*x - d) * 0.5  # 目的関数
constraints = [A@x <= b]  # 制約条件
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", 0.5*cp.norm(C*x - d, p=2).value)
