import numpy as np
import matplotlib.pyplot as plt
m = 0.03
g = 9.81
l = 0.075
eps = 1.1
# l = eps*0.075
kg = -30
gamma = 0.02

c = np.array([[0, 1]])
A = np.array([[0, 1], [g/l, 0]])
b = np.array([[0], [1/(m*l**2)]])
P = np.array([[-30, 0], [0, -30]])
K = np.array([[0.0895, 0.0047]])
# print(P)


def x(x1, x2):
    # x1:角度，x2:角速度
    return np.array([[x1, x2]]).T


def h(x1):
    # 非線形項
    return np.array([[0], [(g/l)*(np.sin(x1)-x1)]])


def u0(x1, x2):
    return -m*g*l*(np.sin(x1)-x1)-K@x(x1, x2)


def z(z1, z2):
    return np.array([[z1, z2]]).T


def z_dot1(x1, x2):
    return A@x(x1, x2)+h(x1)+b*u0(x1, x2)


def z_dot2(x1, x2, z1, z2):
    # 拡張ISMC
    return A@x(x1, x2)+h(x1)+b*u0(x1, x2)-P@(x(x1, x2)-z(z1, z2))


def x_dot1(x1, x2, z1, z2):
    return A@x(x1, x2)+h(x1)+b*u1(x1, x2, z1, z2)


def x_dot2(x1, x2, z1, z2):
    return A@x(x1, x2)+h(x1)+b*u2(x1, x2, z1, z2)


def S(x1, x2, z1, z2):
    # 切換関数(滑り面)
    return c@(x(x1, x2)-z(z1, z2))


# def S_dot(x1, x2, z1, z2):
#     return c@(x_dot1(x1, x2, z1, z2)-z_dot1(x1, x2))


def u1(x1, x2, z1, z2):
    return u0(x1, x2)-gamma*np.sign(S(x1, x2, z1, z2))+np.linalg.inv(c@b)*kg*S(x1, x2, z1, z2)

# def u1(x1, x2, z1, z2):
#     return u0(x1, x2)-gamma*np.sign(S(x1, x2, z1, z2))+(1/(c@b))*kg*S(x1, x2, z1, z2)


def u2(x1, x2, z1, z2):
    # 拡張ISMCの制御入力
    return u0(x1, x2)-gamma*np.sign(S(x1, x2, z1, z2))


# print(type(b@u0(np.pi/2, 0)))
# print(type(x_dot(0, 0, 0, 0)))
# x1 = np.pi/6
# x2 = 0
# z1 = np.pi/6
# z2 = 0
x_cal = np.array([[np.pi/6], [0]])
z_cal = np.array([[np.pi/6], [0]])
x_cal2 = np.array([[np.pi/6], [0]])
z_cal2 = np.array([[np.pi/6], [0]])
# S_cal = S(x_cal[0, 0], x_cal[1, 0], z_cal[0, 0], z_cal[1, 0])
T = 1  # シミュレーション時間
n = 1000  # 分割数
dt = T/n  # 分割時間
t = 0
t_list = np.linspace(0, T, n+1)
# print(t_list)
u0_list = np.zeros((1, len(t_list)))
S_list = np.zeros((1, len(t_list)))
u_list = np.zeros((1, len(t_list)))
x_list = np.zeros((2, len(t_list)))
u0_list2 = np.zeros((1, len(t_list)))
S_list2 = np.zeros((1, len(t_list)))
u_list2 = np.zeros((1, len(t_list)))
x_list2 = np.zeros((2, len(t_list)))
# x_list[:, 0] = [np.pi/6, 0]
z_list = np.zeros((2, len(t_list)))
z_list2 = np.zeros((2, len(t_list)))
# z_list[:, 0] = [np.pi/6, 0]

# print(type(z_list))
# 'numpy.ndarray' object is not callable
# 関数名と関数の戻り値が保存される変数名が同じ場合、エラーが表示されることがあります。

# 従来のISMC
for i in range(0, len(t_list)):
    # S = S(x_cal[0, 0], x_cal[1, 0], z_cal[0, 0], z_cal[1, 0])
    u0_list[0, i] = u0(x_cal[0, 0], x_cal[1, 0])
    u_list[0, i] = u1(x_cal[0, 0], x_cal[1, 0], z_cal[0, 0], z_cal[1, 0])
    S_list[0, i] = S(x_cal[0, 0], x_cal[1, 0], z_cal[0, 0], z_cal[1, 0])
    z_list[0, i] = z_cal[0, 0]
    z_list[1, i] = z_cal[1, 0]
    x_list[0, i] = x_cal[0, 0]
    x_list[1, i] = x_cal[1, 0]

    z_cal = z_cal+z_dot1(x_cal[0, 0], x_cal[1, 0])*dt
    x_cal = x_cal+x_dot1(x_cal[0, 0],
                         x_cal[1, 0], z_cal[0, 0], z_cal[1, 0])*dt
# 拡張ISMC
    u0_list2[0, i] = u0(x_cal2[0, 0], x_cal2[1, 0])
    u_list2[0, i] = u2(x_cal2[0, 0], x_cal2[1, 0], z_cal2[0, 0], z_cal2[1, 0])
    S_list2[0, i] = S(x_cal2[0, 0], x_cal2[1, 0], z_cal2[0, 0], z_cal2[1, 0])
    z_list2[0, i] = z_cal2[0, 0]
    z_list2[1, i] = z_cal2[1, 0]
    x_list2[0, i] = x_cal2[0, 0]
    x_list2[1, i] = x_cal2[1, 0]

    z_cal2 = z_cal2+z_dot2(x_cal2[0, 0], x_cal2[1, 0],
                           z_cal2[0, 0], z_cal2[1, 0])*dt
    x_cal2 = x_cal2+x_dot2(x_cal2[0, 0],
                           x_cal2[1, 0], z_cal2[0, 0], z_cal2[1, 0])*dt

    # t = t+dt
# print(x_list[0, :])
# print(S_list)
# print(u_list[0, :])

plt.figure()
plt.plot(t_list, x_list[0, :]*(180/np.pi))
plt.plot(t_list, x_list2[0, :]*(180/np.pi))
plt.ylabel("Angle[deg]")
plt.xlabel("Time[sec]")
plt.show()

plt.figure()
plt.plot(t_list, S_list[0, :])
plt.plot(t_list, S_list2[0, :])
plt.ylabel("Switching function")
plt.xlabel("Time[sec]")
plt.show()

plt.figure()
plt.plot(t_list, u_list[0, :])
plt.plot(t_list, u_list2[0, :])
plt.ylabel("Control input[Nm]")
plt.xlabel("Time[sec]")
plt.show()

plt.figure()
plt.plot(t_list, u0_list[0, :])
plt.plot(t_list, u0_list2[0, :])
plt.ylabel("Nominal Control input[Nm]")
plt.xlabel("Time[sec]")
plt.show()
