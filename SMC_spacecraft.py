import numpy as np
import matplotlib.pyplot as plt

# Jは慣性テンソル
J = np.array([[420, 18, -15],
              [18, 256, -12],
              [-15, -12, 618]])

rho = 0.8
mu = 20
alpha1 = np.array([[7.53, 0, 0],
                   [0, 5.12, 0],
                   [0, 0, 8.99]])

alpha2 = np.array([[0.37, 0, 0],
                   [0, 0.16, 0],
                   [0, 0, 0.56]])

alpha3 = np.array([[2.14, 0, 0],
                   [0, 3.45, 0],
                   [0, 0, 2.01]])

alpha4 = np.array([[1.24, 0, 0],
                   [0, 2.51, 0],
                   [0, 0, 1.86]])

# 角速度


def omega(omega1, omega2, omega3):
    return np.array([[omega1],
                     [omega2],
                     [omega3]])

# MRP


def sigma(sigma1, sigma2, sigma3):
    return np.array([[sigma1],
                     [sigma2],
                     [sigma3]])


def omega_cross(omega1, omega2, omega3):
    return np.array([[0, -omega3, omega2],
                     [omega3, 0, -omega1],
                     [-omega2, omega1, 0]])


def sigma_cross(sigma1, sigma2, sigma3):
    return np.array([[0, -sigma3, sigma2],
                     [sigma3, 0, -sigma1],
                     [-sigma2, sigma1, 0]])


def G(sigma1, sigma2, sigma3):
    return ((1-sigma(sigma1, sigma2, sigma3).T@sigma(sigma1, sigma2, sigma3))*np.eye(3)+2*sigma_cross(sigma1, sigma2, sigma3)+2*sigma(sigma1, sigma2, sigma3)@sigma(sigma1, sigma2, sigma3).T)/4


def M(sigma1, sigma2, sigma3):
    return (np.sqrt(rho)/(4*(1+sigma(sigma1, sigma2, sigma3).T@sigma(sigma1, sigma2, sigma3))))*((1-sigma(sigma1, sigma2, sigma3).T@sigma(sigma1, sigma2, sigma3))*np.eye(3)+2*sigma_cross(sigma1, sigma2, sigma3))

# 外乱


def d(t):
    return np.array([[(2+2*np.sin((np.pi/75)*t))*10**(-3)],
                     [(1+3*np.cos((np.pi/75)*t))*10**(-3)],
                     [(3+2*np.sin((np.pi/75)*t))*10**(-3)]])

# 滑り面


def S(omega1, omega2, omega3, sigma1, sigma2, sigma3):
    return omega(omega1, omega2, omega3)+(np.sqrt(rho)*sigma(sigma1, sigma2, sigma3)/(1+sigma(sigma1, sigma2, sigma3).T@sigma(sigma1, sigma2, sigma3)))


omega_cal = np.array([[0],
                      [0],
                      [0]])

sigma_cal = np.array([[0.3],
                      [0.2],
                      [-0.3]])

omega0_cal = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

sigma0_cal = np.array([[0, 0.3, 0.2],
                       [-0.3, 0, 0.3],
                       [-0.2, 0.3, 0]])

S_cal = S(0, 0, 0, 0.3, 0.2, -0.3)

omega1 = 0
omega2 = 0
omega3 = 0
sigma1 = 0.3
sigma2 = 0.2
sigma3 = -0.3
# omega = omega(0, 0, 0)
# sigma = sigma(0.3, 0.2, -0.3)
# omega_cross = omega_cross(0, 0, 0)
# sigma_cross = sigma_cross(0.3, 0.2, -0.3)
# S = S(0, 0, 0, 0.3, 0.2, -0.3)
T = 250  # シミュレーション時間
n = 250  # 分割数
dt = T/n  # 分割時間
t = 0
t_list = np.linspace(0, T, n+1)
# print(t_list)
usmc = np.zeros((3, len(t_list)))
# usmc = usmc.T
# usmc = []
# print(usmc.shape)
omega_list = np.zeros((3, len(t_list)))
sigma_list = np.zeros((3, len(t_list)))
# sigma_list[0, 0] = 0.3
# sigma_list[1, 0] = 0.2
# sigma_list[2, 0] = -0.3
sigma_list[:, 0] = [0.3, 0.2, -0.3]
# print(sigma_list)
# print(type(usmc))
# print(sigma_list)
for i in range(0, len(t_list)):
    # 外乱なし
    # u = omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal-J@M(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]
    #                                                                                    )@omega_cal-alpha1@S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0])-alpha2@np.sign(S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]))

    # 符号関数の場合
    u = omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal-d(t)-J@M(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]
                                                                                            )@omega_cal-alpha1@S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0])-alpha2@np.sign(S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]))

    # tanhの場合
    # u = omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal-d(t)-J@M(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]
    #                                                                                         )@omega_cal-alpha3@S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0])-alpha4@np.tanh(mu*S(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0], sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]))

    omega_list[0, i] = omega_cal[0, 0]
    omega_list[1, i] = omega_cal[1, 0]
    omega_list[2, i] = omega_cal[2, 0]

    omega_cal = omega_cal+(-(np.linalg.inv(J))@omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal +
                           (np.linalg.inv(J))@u+(np.linalg.inv(J))@d(t))*dt

    # omega_cal = omega(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])+(-(np.linalg.inv(J))@omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal +
    #                                                                       (np.linalg.inv(J))@u+(np.linalg.inv(J))@d(t))*dt

    # 外乱なし
    # omega_cal = omega_cal+(-(np.linalg.inv(J))@omega_cross(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0])@J@omega_cal +
    #                        (np.linalg.inv(J))@u)*dt

    sigma_list[0, i] = sigma_cal[0, 0]
    sigma_list[1, i] = sigma_cal[1, 0]
    sigma_list[2, i] = sigma_cal[2, 0]
    # 文字を別に置かない場合
    sigma_cal = sigma(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]) + \
        (G(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]) @
         omega(omega_cal[0, 0], omega_cal[1, 0], omega_cal[2, 0]))*dt

    # 文字を別に置いた場合
    # sigma_cal = sigma_cal + (G(sigma_cal[0, 0], sigma_cal[1, 0], sigma_cal[2, 0]) @
    #                          omega_cal)*dt
    # sigma_cal = sigma_cal-np.sqrt(rho)/4*sigma_cal

    usmc[0, i] = u[0, 0]
    usmc[1, i] = u[1, 0]
    usmc[2, i] = u[2, 0]

    t = t+dt
#     omega_list.append(omega)
#     sigma_list.append(sigma)
# print(usmc[0, :])
# print(sigma_list[1, :])

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_list, usmc[0, :])
# plt.ylabel("u1(Nm)")
plt.subplot(3, 1, 2)
plt.plot(t_list, usmc[1, :])
# plt.ylabel("u2(Nm)")
plt.ylabel("Control torque $\mathbf{u}$(Nm)")
plt.subplot(3, 1, 3)
plt.plot(t_list, usmc[2, :])
plt.xlabel("Time(s)")
# plt.ylabel("u3(Nm)")
plt.show()

plt.figure()
# plt.subplot(3, 1, 1)
plt.plot(t_list, sigma_list[0, :], label="$\sigma_1$")
# plt.subplot(3, 1, 2)
plt.plot(t_list, sigma_list[1, :], label="$\sigma_2$")
# plt.subplot(3, 1, 3)
plt.plot(t_list, sigma_list[2, :], label="$\sigma_3$")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("$\mathbf{\sigma}$")
plt.show()

plt.figure()
# plt.subplot(3, 1, 1)
plt.plot(t_list, omega_list[0, :]/10**(-2), label="$\omega_1$")
# plt.subplot(3, 1, 2)
plt.plot(t_list, omega_list[1, :]/10**(-2), label="$\omega_2$")
# plt.subplot(3, 1, 3)
plt.plot(t_list, omega_list[2, :]/10**(-2), label="$\omega_3$")
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Angular Velocities $\mathbf{\omega}$(rad/s)")
plt.show()

# print(d(0))
# print(omega_cross(1, 1, 1))
# print(np.sign(S(0, 0, 0, 0, 0, 0)).shape)
# print(S(0, 0, 0, 0, 0, 0).shape)
# print(np.linalg.inv(J))
