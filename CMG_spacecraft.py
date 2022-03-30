import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
#from numpy.lib.function_base import kaiser

J = np.array([[420, 18, -15],
              [18, 256, -12],
              [-15, -12, 618]])
omega1 = 0
omega2 = 0
omega3 = 0

# omega_0=np.array([[omega1],
#                     [omega2],
#                     [omega3]])

sigma1 = 0.3
sigma2 = 0.2
sigma3 = -0.3

sigma_0 = np.array([[0.3],
                    [0.2],
                    [-0.3]])
ro = 0.8
mu = 20
alpha_1 = np.array([[7.53, 0, 0],
                    [0, 5.12, 0],
                    [0, 0, 8.99]])

alpha_2 = np.array([[0.37, 0, 0],
                    [0, 0.16, 0],
                    [0, 0, 0.56]])

alpha_3 = np.array([[2.14, 0, 0],
                    [0, 3.45, 0],
                    [0, 0, 2.01]])

alpha_4 = np.array([[1.24, 0, 0],
                    [0, 2.51, 0],
                    [0, 0, 1.86]])

T = 25000
#n = 10
dt = 0.01

t_list = np.linspace(0, T, T+1)
u_list = np.zeros((3, len(t_list)))
omega_list = np.zeros((3, len(t_list)))
sigma_list = np.zeros((3, len(t_list)))
d_list = np.zeros((3, len(t_list)))


def M(sigma1, sigma2, sigma3):
    sigma = np.array([[sigma1], [sigma2], [sigma3]])
    sigma_cross = np.array([[0, -sigma3, sigma2],
                            [sigma3, 0, -sigma1],
                            [-sigma2, sigma1, 0]])
    return np.sqrt(ro)*((1-sigma.T@sigma)*np.eye(3)+2*sigma_cross)/(4*(1+sigma.T@sigma))


def G_sigma(sigma1, sigma2, sigma3):
    sigma = np.array([[sigma1], [sigma2], [sigma3]])
    sigma_cross = np.array([[0, -sigma3, sigma2],
                            [sigma3, 0, -sigma1],
                            [-sigma2, sigma1, 0]])
    return ((1-sigma.T@sigma)*np.eye(3)+2*sigma_cross+2*sigma@sigma.T)/4


def sigma(G_sigma, omega1, omega2, omega3):
    omega = np.array([[omega1], [omega2], [omega3]])
    return G_sigma@omega


def u(omega1, omega2, omega3, M, s1, s2, s3, d):
    omega = np.array([[omega1], [omega2], [omega3]])
    s = np.array([[s1], [s2], [s3]])
    omega_cross = np.array([[0, -omega3, omega2],
                            [omega3, 0, -omega1],
                            [-omega2, omega1, 0]])
    sgn_s = np.array([[np.sign(s1)], [np.sign(s2)], [np.sign(s3)]])
    return omega_cross@J@omega-J@M@omega-alpha_1@s-alpha_2@sgn_s-d


def u2(omega1, omega2, omega3, M, s1, s2, s3, d):
    omega = np.array([[omega1], [omega2], [omega3]])
    s = np.array([[s1], [s2], [s3]])
    omega_cross = np.array([[0, -omega3, omega2],
                            [omega3, 0, -omega1],
                            [-omega2, omega1, 0]])
    tanh = np.array([[np.tanh(mu*s1)], [np.tanh(mu*s2)], [np.tanh(mu*s3)]])
    return omega_cross@J@omega-J@M@omega-alpha_3@s-alpha_4@tanh-d


def d(t):
    return np.array([[2+2*np.sin(np.pi*t/75)],
                     [1+3*np.cos(np.pi*t/75)],
                     [3+2*np.sin(np.pi*t/75)]])/10**3


def omega(omega1, omega2, omega3, u, d):
    omega = np.array([[omega1], [omega2], [omega3]])
    omega_cross = np.array([[0, -omega3, omega2],
                            [omega3, 0, -omega1],
                            [-omega2, omega1, 0]])
    return np.linalg.inv(J)@(-omega_cross@J@omega+u+d)


def s(omega1, omega2, omega3, sigma1, sigma2, sigma3):
    omega = np.array([[omega1], [omega2], [omega3]])
    sigma = np.array([[sigma1], [sigma2], [sigma3]])
    return omega+np.sqrt(ro)*sigma/(1+sigma.T@sigma)


for i in range(0, len(t_list)):
    t = (i+1)*dt
    d_now = d(t)
    # d=d_now
    d_list[0, i] = d_now[0, 0]
    d_list[1, i] = d_now[1, 0]
    d_list[2, i] = d_now[2, 0]

    s_now = s(omega1, omega2, omega3, sigma1,
              sigma2, sigma3)  # .reshape([3, 1])
    # print(s_now)
    s1 = s_now[0, 0]
    s2 = s_now[1, 0]
    s3 = s_now[2, 0]

    M_now = M(sigma1, sigma2, sigma3)
    #M = M_now

    # u_now = u(omega1, omega2, omega3, M_now, s1, s2, s3, d_now)
    #u = u_now
    # u_list[0, i] = u_now[0, 0]  # sgnの場合
    # u_list[1, i] = u_now[1, 0]
    # u_list[2, i] = u_now[2, 0]

    u_now = u2(omega1, omega2, omega3, M_now, s1, s2, s3, d_now)
    # #u = u_now
    u_list[0, i] = u_now[0, 0]  # tanhの場合
    u_list[1, i] = u_now[1, 0]
    u_list[2, i] = u_now[2, 0]
    # # print(u_now)

    G_sigma_now = G_sigma(sigma1, sigma2, sigma3)
    #G_sigma = G_sigma_now

    sigma_now = sigma(G_sigma_now, omega1, omega2, omega3)  # .reshape([3, 1])
    sigma_list[0, i] = sigma1
    sigma_list[1, i] = sigma2
    sigma_list[2, i] = sigma3
    sigma1 = sigma1+sigma_now[0, 0]*dt
    sigma2 = sigma2+sigma_now[1, 0]*dt
    sigma3 = sigma3+sigma_now[2, 0]*dt

    omega_now = omega(omega1, omega2, omega3, u_now, d_now)  # .reshape([3,1])
    omega_list[0, i] = omega1
    omega_list[1, i] = omega2
    omega_list[2, i] = omega3
    omega1 = omega1+omega_now[0, 0]*dt
    omega2 = omega2+omega_now[1, 0]*dt
    omega3 = omega3+omega_now[2, 0]*dt

    # print(omega_now)

# print(sigma_now.shape)
# print((sigma_now.T@sigma_now).shape)
# print(u_now.shape)
# print(omega_now.shape)

# print(type(u_now))
# print(u_list)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_list/100, u_list[0, :])
plt.subplot(3, 1, 2)
plt.plot(t_list/100, u_list[1, :])
plt.subplot(3, 1, 3)
plt.plot(t_list/100, u_list[2, :])
# plt.plot(t_list, u_list[1, :])
# plt.plot(t_list, u_list[2, :])
plt.figure()
plt.subplot()
plt.plot(t_list/100, sigma_list[0, :])
plt.plot(t_list/100, sigma_list[1, :])
plt.plot(t_list/100, sigma_list[2, :])

plt.figure()
plt.plot(t_list/100, omega_list[0, :]/10**(-2))
plt.plot(t_list/100, omega_list[1, :]/10**(-2))
plt.plot(t_list/100, omega_list[2, :]/10**(-2))


# fig, ax = plt.subplots()
# ax[0].plot(t_list, omega_list[0, :])
# # ax[1].plot(t_list, omega_list[1 :])
# # ax[2].plot(t_list, omega_list[2, :])
# ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(style="sci",  axis="y", scilimits=(0, 0))
# fig,ay=plt.subplots(2,1,2)
# ay=plot(t_list, omega_list[1, :])
# ay.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
# ay.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))

plt.show()


# for i in range(0,len(t_list)):
#     x_dt=x+u
#     x=x+x_dt*dt
#     x_list[0,i]=x[0,0]
#     x_list[1,i]=x[1,0]
#     x_list[2,i]=x[2,0]
#     u=x+np.ones((3,1))
#     u_list[0,i]=u[0,0]
#     u_list[1,i]=u[1,0]
#     u_list[2,i]=u[2,0]

# print(x_list)
# print(u_list)
