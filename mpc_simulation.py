# -*- coding: utf-8 -*-

# Theme : Non-Linear Model Predictive Control
# Reference : https://myenigma.hatenablog.com/entry/2016/07/25/214014

import time
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
print("Simulation start")

np.random.seed(1)
n = 4
m = 2
T = 50

alpha = 0.2
beta = 5.0

A = np.eye(n) + alpha*np.random.randn(n, n)
B = np.random.randn(n, m)
x_0 = beta*np.random.randn(n)

zero_x = np.array([0, 0, 0, 0])

x = Variable([n, T+1])
u = Variable([m, T])

# u_1 = np.zeros(T)
# u_2 = np.zeros(T)

states = []
constr = [x[:, T] == zero_x, x[:, 0] == x_0]
for t in range(T):
    cost = sum_squares(x[:, t+1]) + sum_squares(u[:, t])
    constr += [x[:, t+1] == A@x[:, t] + B@u[:, t], norm(u[:, t], 'inf') <= 1]
    # constr += [x[:,T] == zero_x, x[:,0] == x_0]
    states.append(Problem(Minimize(cost), constr))
prob = sum(states)

start = time.time()
result = prob.solve(verbose=True)
elapsed_time = time.time() - start
# print("calc time:{0}".format(elapsed_time)) + "[sec]"
print("calc time:{0}".format(elapsed_time) + "[sec]")
print("Information about : x")
print(x.value)
print("Information about : u")
print(u.value)

if result == float("inf"):
    print("Cannot optimize")
    import sys
    sys.exit()

f = plt.figure()
ax = f.add_subplot(211)
# u1 = np.array(u[0,:].value[0,:])[0].tolist()
u1 = np.array(u[0, :].value)
# u2 = np.array(u[1,:].value[0,:])[0].tolist()
u2 = np.array(u[1, :].value)
plt.plot(u1, '-r', label="u1")
plt.plot(u2, '-b', label="u2")
plt.ylabel(r"$u_t$", fontsize=16)
plt.yticks(np.linspace(-1.0, 1.0, 3))
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
# x1 = np.array(x[0,:].value[0,:])[0].tolist()
x1 = np.array(x[0, :].value)
# x2 = np.array(x[1,:].value[0,:])[0].tolist()
x2 = np.array(x[1, :].value)
# x3 = np.array(x[2,:].value[0,:])[0].tolist()
x3 = np.array(x[2, :].value)
# x4 = np.array(x[3,:].value[0,:])[0].tolist()
x4 = np.array(x[3, :].value)
plt.plot(range(T+1), x1, '-r', label="x1")
plt.plot(range(T+1), x2, '-b', label="x2")
plt.plot(range(T+1), x3, '-g', label="x3")
plt.plot(range(T+1), x4, '-k', label="x4")
plt.yticks([-25, 0, 25])
plt.ylim([-25, 25])
plt.ylabel(r"$x_t$", fontsize=16)
plt.xlabel(r"$t$", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save Figure
# f.savefig("mpc_result.pdf")
