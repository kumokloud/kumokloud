from control import matlab
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# 伝達関数の作成:matlab.tf([分子(num)],[分母(den)])
Np = [1]
Dp = [0.1, 1.1, 1, 0]
Dp2 = [1, 1, 0]
G = matlab.tf(Np, Dp)
G_hat = matlab.tf(Np, Dp2)
t = np.linspace(0, 5, 501)
# print(G)
# print(G_hat)
# ボード線図作成:matlab.bode(伝達関数)
plt.figure()
matlab.bode(G)
matlab.bode(G_hat)
plt.show()
# ナイキスト線図の作成
plt.figure()
matlab.nyquist(G)
matlab.nyquist(G_hat)
plt.show()
# こういう書き方らしい:インパルス関数の場合:matlab.impulse(),ステップ関数の場合:matlab.step()
u1, T = matlab.impulse(G, t)
u2, T = matlab.impulse(G_hat, t)
gain1 = matlab.margin(G)
gain2 = matlab.margin(G_hat)
# print(gain1)  # 返り値:ゲイン余裕(dB)，位相余裕(deg)，ゲイン余裕の周波数(rad/sec)，位相余裕の周波数(rad/sec)
# print(gain2)
plt.figure()
plt.grid()
plt.plot(T, u1, label="G")
plt.plot(T, u2, label="G_hat")
plt.legend()  # グラフの凡例
plt.show()
# ブロック線図(直列)
G_G_hat = matlab.series(G, G_hat)
print(G_G_hat)
# ブロック線図(並列)
G_G_hat2 = matlab.parallel(G, G_hat)
print(G_G_hat2)
# ブロック線図(フィードバック)
G_G_hat3 = matlab.feedback(G, G_hat)
print(G_G_hat3)
