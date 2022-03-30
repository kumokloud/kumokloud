from control import matlab
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# 伝達関数の作成:matlab.tf([分子(num)],[分母(den)])
K = 0.5
Np = [K]
Dp = [0.1, 1.1, 1, K]
G = matlab.tf(Np, Dp)
# step_func = matlab.tf([0, 1], [1, 0])
t = np.linspace(0, 15, 501)
print(G)
# 0次ホールドによる離散化
ts1 = 0.1  # サンプリング時間
ts2 = 0.5
ts3 = 1.0
G1_1 = matlab.c2d(G, ts1, method="zoh")
G1_2 = matlab.c2d(G, ts2, method="zoh")
G1_3 = matlab.c2d(G, ts3, method="zoh")
print("離散時間システム(zoh)", G1_1)
print("離散時間システム(zoh)", G1_2)
print("離散時間システム(zoh)", G1_3)
# 双一次変換による離散化
G2_1 = matlab.c2d(G, ts1, method="tustin")
G2_2 = matlab.c2d(G, ts2, method="tustin")
G2_3 = matlab.c2d(G, ts3, method="tustin")
print("離散化時間システム(tustin)", G2_1)
print("離散化時間システム(tustin)", G2_2)
print("離散化時間システム(tustin)", G2_3)
# ボード線図作成:matlab.bode(伝達関数)
# plt.figure()
# matlab.bode(G)
# plt.show()
# ナイキスト線図の作成
# plt.figure()
# matlab.nyquist(G)
# plt.show()
# こういう書き方らしい:インパルス関数の場合:matlab.impulse(),ステップ関数の場合:matlab.step()
t1 = np.linspace(0, 15, int(15/ts1)+1)
t2 = np.linspace(0, 15, int(15/ts2)+1)
t3 = np.linspace(0, 15, int(15/ts3)+1)
# step, T = matlab.step(step_func, t)
u, T = matlab.step(G, t)
u1, T1 = matlab.step(G1_1, t1)
u2, T2 = matlab.step(G1_2, t2)
u3, T3 = matlab.step(G1_3, t3)
# 応答のグラフ化
plt.figure()
plt.grid()
# plt.plot(T, step, ls='')
plt.plot(T, u, label="G")
plt.plot(T1, u1, label="G1_1")
plt.plot(T2, u2, label="G1_2")
plt.plot(T3, u3, label="G1_3")
plt.xticks([0, 5, 10, 15])
plt.yticks([0, 0.5, 1, 1.5])
plt.legend()  # グラフの凡例
plt.show()
