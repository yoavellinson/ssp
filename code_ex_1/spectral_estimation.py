########################################################################
#                      Ssp 1 - Programing Assigment                   
#   File Name: Spectral_estimation.py
#   Content: main function and plots
#
#   Name: Yoav Ellinson                           Name: Chen Yaakov Sror
#   Id: 206036949                                 Id:   203531645
########################################################################

########################################################################
                        # Defines and Includes
########################################################################
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm


########################################################################
#                            Question 1
########################################################################

# Calculating the spectrum for x_1 and x_2 with the results of the 
# analytic calculation

def Sxx1(w):
    return 1 + (9/13)*np.cos(w) - (4/13)*np.cos(2*w)

def Sxx2(w):
    return 0.86/(np.abs(1-0.7*np.exp(-1j*w))**2)

n = 2049
w = np.linspace(0, np.pi, num=n)

########################################################################
#                            Question 2
########################################################################
# These values were calculated analyticly

sigma_w1 = np.sqrt(1/26)
sigma_w2 = np.sqrt(0.86)

n1 = 1024
n2 = 2048

def gen_signals(sigma_w1=np.sqrt(1/26), n1=1024, sigma_w2=np.sqrt(0.86), n2=1024):  # The default values are for the question
    #   Generating n1 samples from a normal distribution by the x1 equation
    w1 = np.random.normal(0, sigma_w1, n1)                                          
    x1 = np.array([w1[n]-3*w1[n-1]-4*w1[n-2] for n in range(n1)])   
    w2 = np.random.normal(0, sigma_w2, n2)

    # A Function to calculate the signal x2
    def gen_x2(x2, n):
        return 0.7*x2[n-1] + w2[n]
    x2_tmp = np.zeros(n2)
    x2_tmp[-1] = -0.5880857651749811    #initial value - explained in the report
    for n in range(1, n2):
        x2_tmp[n] = gen_x2(x2_tmp, n)

    return x1, x2_tmp

########################################################################
# This is the same function as gen signal but without initial values
########################################################################
# def gen_signals_old(sigma_w1=np.sqrt(1/26), n1=1024, sigma_w2=np.sqrt(0.86), n2=2048):  # for the question
#     w1 = np.random.normal(0, sigma_w1, n1)
#     x1 = np.array([w1[n]-3*w1[n-1]-4*w1[n-2] for n in range(n1)])

#     w2 = np.random.normal(0, sigma_w2, n2)

#     def gen_x2(x2, n):
#         return 0.7*x2[n-1] + w2[n]
#     x2_tmp = np.zeros(n2)
#     for n in range(1, n2):
#         x2_tmp[n] = gen_x2(x2_tmp, n)

#     x2 = x2_tmp[-1024:]
#     return x1, x2


# Creating matricies that will hold the Monte Carlo result
# Every Sx1/Sx2 Element will have Mc Rows and 2049 Cols.
Mc = 100

Sx1_per = np.empty((Mc, 2049))
Sx1_cor = np.empty((Mc, 2049))
Sx1_bartlet_64 = np.empty((Mc, 2049))
Sx1_bartlet_16 = np.empty((Mc, 2049))
Sx1_welch_64 = np.empty((Mc, 2049))
Sx1_welch_16 = np.empty((Mc, 2049))
Sx1_bt_4 = np.empty((Mc, 2049))
Sx1_bt_2 = np.empty((Mc, 2049))

Sx2_per = np.empty((Mc, 2049))
Sx2_cor = np.empty((Mc, 2049))
Sx2_bartlet_64 = np.empty((Mc, 2049))
Sx2_bartlet_16 = np.empty((Mc, 2049))
Sx2_welch_64 = np.empty((Mc, 2049))
Sx2_welch_16 = np.empty((Mc, 2049))
Sx2_bt_4 = np.empty((Mc, 2049))
Sx2_bt_2 = np.empty((Mc, 2049))

Sx1_analytic = Sxx1(w)
Sx2_analytic = Sxx2(w)

########################################################################
# Generating all the data
########################################################################

for i in tqdm(range(Mc)):
    x1, x2 = gen_signals()
    Sx1_per[i], _ = calc_periodegram(x1)
    Sx1_cor[i], _ = calc_correlogram(x1)
    Sx1_bartlet_64[i], _ = calc_bartlet(x1, 64)
    Sx1_bartlet_16[i], _ = calc_bartlet(x1, 16)
    Sx1_welch_64[i], _ = calc_welch(x1, k=61, L=64, D=48)
    Sx1_welch_16[i], _ = calc_welch(x1, k=253, L=16, D=12)
    Sx1_bt_4[i], _ = calc_blackman_tukey(x1, L=4)
    Sx1_bt_2[i], _ = calc_blackman_tukey(x1, L=2)
    Sx2_per[i], _ = calc_periodegram(x2)
    Sx2_cor[i], _ = calc_correlogram(x2)
    Sx2_bartlet_64[i], _ = calc_bartlet(x2, 64)
    Sx2_bartlet_16[i], _ = calc_bartlet(x2, 16)
    Sx2_welch_64[i], _ = calc_welch(x2, k=61, L=64, D=48)
    Sx2_welch_16[i], _ = calc_welch(x2, k=253, L=16, D=12)
    Sx2_bt_4[i], _ = calc_blackman_tukey(x2, L=4)
    Sx2_bt_2[i], w_M = calc_blackman_tukey(x2, L=2)

########################################################################
# Analysing the Stats
# For x1
########################################################################

Sx1_per_bar, Sx1_per_bias, Sx1_per_var, Sx1_per_mse, Sx1_per_total_bias, Sx1_per_total_var, Sx1_per_total_mse = get_all_stats(
    Sx1_per, Sx1_analytic)
Sx1_cor_bar, Sx1_cor_bias, Sx1_cor_var, Sx1_cor_mse, Sx1_cor_total_bias, Sx1_cor_total_var, Sx1_cor_total_mse = get_all_stats(
    Sx1_cor, Sx1_analytic)
Sx1_bartlet_64_bar, Sx1_bartlet_64_bias, Sx1_bartlet_64_var, Sx1_bartlet_64_mse, Sx1_bartlet_64_total_bias, Sx1_bartlet_64_total_var, Sx1_bartlet_64_total_mse = get_all_stats(
    Sx1_bartlet_64, Sx1_analytic)
Sx1_bartlet_16_bar, Sx1_bartlet_16_bias, Sx1_bartlet_16_var, Sx1_bartlet_16_mse, Sx1_bartlet_16_total_bias, Sx1_bartlet_16_total_var, Sx1_bartlet_16_total_mse = get_all_stats(
    Sx1_bartlet_16, Sx1_analytic)
Sx1_welch_64_bar, Sx1_welch_64_bias, Sx1_welch_64_var, Sx1_welch_64_mse, Sx1_welch_64_total_bias, Sx1_welch_64_total_var, Sx1_welch_64_total_mse = get_all_stats(
    Sx1_welch_64, Sx1_analytic)
Sx1_welch_16_bar, Sx1_welch_16_bias, Sx1_welch_16_var, Sx1_welch_16_mse, Sx1_welch_16_total_bias, Sx1_welch_16_total_var, Sx1_welch_16_total_mse = get_all_stats(
    Sx1_welch_16, Sx1_analytic)
Sx1_bt_4_bar, Sx1_bt_4_bias, Sx1_bt_4_var, Sx1_bt_4_mse, Sx1_bt_4_total_bias, Sx1_bt_4_total_var, Sx1_bt_4_total_mse = get_all_stats(
    Sx1_bt_4, Sx1_analytic)
Sx1_bt_2_bar, Sx1_bt_2_bias, Sx1_bt_2_var, Sx1_bt_2_mse, Sx1_bt_2_total_bias, Sx1_bt_2_total_var, Sx1_bt_2_total_mse = get_all_stats(
    Sx1_bt_2, Sx1_analytic)

########################################################################
# Analysing the Stats
# For x2
########################################################################

Sx2_per_bar, Sx2_per_bias, Sx2_per_var, Sx2_per_mse, Sx2_per_total_bias, Sx2_per_total_var, Sx2_per_total_mse = get_all_stats(
    Sx2_per, Sx2_analytic)
Sx2_cor_bar, Sx2_cor_bias, Sx2_cor_var, Sx2_cor_mse, Sx2_cor_total_bias, Sx2_cor_total_var, Sx2_cor_total_mse = get_all_stats(
    Sx2_cor, Sx2_analytic)
Sx2_bartlet_64_bar, Sx2_bartlet_64_bias, Sx2_bartlet_64_var, Sx2_bartlet_64_mse, Sx2_bartlet_64_total_bias, Sx2_bartlet_64_total_var, Sx2_bartlet_64_total_mse = get_all_stats(
    Sx2_bartlet_64, Sx2_analytic)
Sx2_bartlet_16_bar, Sx2_bartlet_16_bias, Sx2_bartlet_16_var, Sx2_bartlet_16_mse, Sx2_bartlet_16_total_bias, Sx2_bartlet_16_total_var, Sx2_bartlet_16_total_mse = get_all_stats(
    Sx2_bartlet_16, Sx2_analytic)
Sx2_welch_64_bar, Sx2_welch_64_bias, Sx2_welch_64_var, Sx2_welch_64_mse, Sx2_welch_64_total_bias, Sx2_welch_64_total_var, Sx2_welch_64_total_mse = get_all_stats(
    Sx2_welch_64, Sx2_analytic)
Sx2_welch_16_bar, Sx2_welch_16_bias, Sx2_welch_16_var, Sx2_welch_16_mse, Sx2_welch_16_total_bias, Sx2_welch_16_total_var, Sx2_welch_16_total_mse = get_all_stats(
    Sx2_welch_16, Sx2_analytic)
Sx2_bt_4_bar, Sx2_bt_4_bias, Sx2_bt_4_var, Sx2_bt_4_mse, Sx2_bt_4_total_bias, Sx2_bt_4_total_var, Sx2_bt_4_total_mse = get_all_stats(
    Sx2_bt_4, Sx2_analytic)
Sx2_bt_2_bar, Sx2_bt_2_bias, Sx2_bt_2_var, Sx2_bt_2_mse, Sx2_bt_2_total_bias, Sx2_bt_2_total_var, Sx2_bt_2_total_mse = get_all_stats(
    Sx2_bt_2, Sx2_analytic)

########################################################################
#                                Plots
########################################################################

########################################################################
#                         Question 2 - Plots
#                    Correlogram and Periodegram
########################################################################

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
fig.suptitle(
    r'$\bar{S}_{x1}$ and $\bar{S}_{x2}$ Estimations with the Correlogram and Periodegram methods')
ax1.title.set_text(r'$\bar{S}_{x1}$')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$\bar{S}_{x1}(e^{j\omega})$')
ax2.title.set_text(r'$\bar{S}_{x2}$')
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'$\bar{S}_{x2}(e^{j\omega})$')
ax1.plot(w_M, Sx1_per[0], label='Sx1 Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_cor[0], label='Sx1 Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sxx1(w_M), linewidth=1.5, label='Sx1 True($e^{j\omega} $)')

ax2.plot(w_M, Sx2_per[0], label='Sx2 Periodegram ($e^{j\omega} $)')
ax2.plot(w_M, Sx2_cor[0], label='Sx2 Correlogram ($e^{j\omega} $)')
ax2.plot(w_M, Sxx2(w_M), linewidth=1.5, label='Sx2 True ($e^{j\omega} $)')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.savefig('pics/q2.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                               Sx1 bar
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$\bar{S}_{x1}$ Estimation for Mc'+f'={Mc}')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$\bar{S}_{x1}(e^{j\omega})$')
ax1.plot(w_M, Sx1_per_bar,
         label=r'$\bar{S}_{x1}$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_cor_bar,
         label=r'$\bar{S}_{x1}$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_2_bar, label=r'$\bar{S}_{x1}$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_4_bar, label=r'$\bar{S}_{x1}$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_16_bar,
         label=r'$\bar{S}_{x1}$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_64_bar,
         label=r'$\bar{S}_{x1}$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_16_bar,
         label=r'$\bar{S}_{x1}$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_64_bar,
         label=r'$\bar{S}_{x1}$ Welch L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sxx1(w_M), linewidth=1.5,
         label=r'${S}_{x1}$ True($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx1_bar.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                               Sx1 Bias
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$B(e^{j\omega})$ for Mc'+f'={Mc}')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$B_{x1}(e^{j\omega})$')
ax1.plot(w_M, Sx1_per_bias,
         label=r'$B(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_cor_bias,
         label=r'$B(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_2_bias, label=r'$B(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_4_bias, label=r'$B(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_16_bias,
         label=r'$B(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_64_bias,
         label=r'$B(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_16_bias,
         label=r'$B(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_64_bias,
         label=r'$B(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx1_bias.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                            Sx1 Variance
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$Var(e^{j\omega})$ for Mc'+f'={Mc}')
ax1.set_ylabel(r'$V_{x1}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx1_per_var,
         label=r'$V(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_cor_var,
         label=r'$V(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_2_var, label=r'$V(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_4_var, label=r'$V(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_16_var,
         label=r'$V(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_64_var,
         label=r'$V(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_16_var,
         label=r'$V(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_64_var,
         label=r'$V(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx1_var.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                              Sx1 MSE
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$MSE(e^{j\omega})$for Mc'+f'={Mc}')
ax1.set_ylabel(r'$MSE_{x1}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx1_per_mse,
         label=r'$MSE(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_cor_mse,
         label=r'$MSE(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_2_mse,
         label=r'$MSE(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bt_4_mse,
         label=r'$MSE(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_16_mse,
         label=r'$MSE(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_bartlet_64_mse,
         label=r'$MSE(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_16_mse,
         label=r'$MSE(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx1_welch_64_mse,
         label=r'$MSE(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx1_mse.png')
plt.close()


########################################################################
#                         Question 3 - Plots
#                               Sx2 bar
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$\bar{S}_{x2}$ Estimation for Mc'+f'={Mc}')
ax1.set_ylabel(r'$\bar{S}_{x2}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx2_per_bar,
         label=r'$\bar{S}_{x2}$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_cor_bar,
         label=r'$\bar{S}_{x2}$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_2_bar, label=r'$\bar{S}_{x2}$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_4_bar, label=r'$\bar{S}_{x2}$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_16_bar,
         label=r'$\bar{S}_{x2}$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_64_bar,
         label=r'$\bar{S}_{x2}$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_16_bar,
         label=r'$\bar{S}_{x2}$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_64_bar,
         label=r'$\bar{S}_{x2}$ Welch L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sxx2(w_M), linewidth=1.5,
         label=r'${S}_{x2}$ True($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx2_bar.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                               Sx2 Bias
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$B(e^{j\omega})$ for Mc'+f'={Mc}')
ax1.set_ylabel(r'$B_{x2}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx2_per_bias,
         label=r'$B(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_cor_bias,
         label=r'$B(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_2_bias, label=r'$B(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_4_bias, label=r'$B(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_16_bias,
         label=r'$B(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_64_bias,
         label=r'$B(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_16_bias,
         label=r'$B(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_64_bias,
         label=r'$B(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx2_bias.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                            Sx1 Variance
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$Var(e^{j\omega})$ for Mc'+f'={Mc}')
ax1.set_ylabel(r'$V_{x2}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx2_per_var,
         label=r'$V(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_cor_var,
         label=r'$V(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_2_var, label=r'$V(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_4_var, label=r'$V(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_16_var,
         label=r'$V(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_64_var,
         label=r'$V(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_16_var,
         label=r'$V(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_64_var,
         label=r'$V(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx2_var.png')
plt.close()

########################################################################
#                         Question 3 - Plots
#                              Sx1 MSE
########################################################################

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
fig.suptitle(r'$MSE(e^{j\omega})$for Mc'+f'={Mc}')
ax1.set_ylabel(r'$MSE_{x2}(e^{j\omega})$')
ax1.set_xlabel(r'$\omega$')
ax1.plot(w_M, Sx2_per_mse,
         label=r'$MSE(e^{j\omega})$ Periodegram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_cor_mse,
         label=r'$MSE(e^{j\omega})$ Correlogram ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_2_mse,
         label=r'$MSE(e^{j\omega})$ BT L=2 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bt_4_mse,
         label=r'$MSE(e^{j\omega})$ BT L=4 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_16_mse,
         label=r'$MSE(e^{j\omega})$ Bartlet L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_bartlet_64_mse,
         label=r'$MSE(e^{j\omega})$ Bartlet L=64 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_16_mse,
         label=r'$MSE(e^{j\omega})$ Welch L=16 ($e^{j\omega} $)')
ax1.plot(w_M, Sx2_welch_64_mse,
         label=r'$MSE(e^{j\omega})$ Welch L=64 ($e^{j\omega} $)')
plt.legend()
plt.savefig('pics/q3_sx2_mse.png')
plt.close()

########################################################################
#                             Question 4 
########################################################################

x1_B_data = {'Sx1 Periodegram': Sx1_per_total_bias,
             'Sx1 Correlogram': Sx1_cor_total_bias,
             'Sx1 BT_2': Sx1_bt_2_total_bias,
             'Sx1_BT_4': Sx1_bt_4_total_bias,
             'Sx1 Bartlet_16': Sx1_bartlet_16_total_bias,
             'Sx1 Bartlet_64': Sx1_bartlet_64_total_bias,
             'Sx1 Welch_16': Sx1_welch_16_total_bias,
             'Sx1 Welch_64': Sx1_welch_64_total_bias}

x1_Var_data = {'Sx1 Periodegram': Sx1_per_total_var,
               'Sx1 Correlogram': Sx1_cor_total_var,
               'Sx1 BT_2': Sx1_bt_2_total_var,
               'Sx1_BT_4': Sx1_bt_4_total_var,
               'Sx1 Bartlet_16': Sx1_bartlet_16_total_var,
               'Sx1 Bartlet_64': Sx1_bartlet_64_total_var,
               'Sx1 Welch_16': Sx1_welch_16_total_var,
               'Sx1 Welch_64': Sx1_welch_64_total_var}

x1_MSE_data = {'Sx1 Periodegram': Sx1_per_total_mse,
               'Sx1 Correlogram': Sx1_cor_total_mse,
               'Sx1 BT_2': Sx1_bt_2_total_mse,
               'Sx1_BT_4': Sx1_bt_4_total_mse,
               'Sx1 Bartlet_16': Sx1_bartlet_16_total_mse,
               'Sx1 Bartlet_64': Sx1_bartlet_64_total_mse,
               'Sx1 Welch_16': Sx1_welch_16_total_mse,
               'Sx1 Welch_64': Sx1_welch_64_total_mse}

x2_B_data = {'Sx2 Periodegram': Sx2_per_total_bias,
             'Sx2 Correlogram': Sx2_cor_total_bias,
             'Sx2 BT_2': Sx2_bt_2_total_bias,
             'Sx2_BT_4': Sx2_bt_4_total_bias,
             'Sx2 Bartlet_16': Sx2_bartlet_16_total_bias,
             'Sx2 Bartlet_64': Sx2_bartlet_64_total_bias,
             'Sx2 Welch_16': Sx2_welch_16_total_bias,
             'Sx2 Welch_64': Sx2_welch_64_total_bias}

x2_Var_data = {'Sx2 Periodegram': Sx2_per_total_var,
               'Sx2 Correlogram': Sx2_cor_total_var,
               'Sx2 BT_2': Sx2_bt_2_total_var,
               'Sx2_BT_4': Sx2_bt_4_total_var,
               'Sx2 Bartlet_16': Sx2_bartlet_16_total_var,
               'Sx2 Bartlet_64': Sx2_bartlet_64_total_var,
               'Sx2 Welch_16': Sx2_welch_16_total_var,
               'Sx2 Welch_64': Sx2_welch_64_total_var}

x2_MSE_data = {'Sx2 Periodegram': Sx2_per_total_mse,
               'Sx2 Correlogram': Sx2_cor_total_mse,
               'Sx2 BT_2': Sx2_bt_2_total_mse,
               'Sx2_BT_4': Sx2_bt_4_total_mse,
               'Sx2 Bartlet_16': Sx2_bartlet_16_total_mse,
               'Sx2 Bartlet_64': Sx2_bartlet_64_total_mse,
               'Sx2 Welch_16': Sx2_welch_16_total_mse,
               'Sx2 Welch_64': Sx2_welch_64_total_mse}

########################################################################
#                         Question 4 - Plots
#                         Average Parameters
########################################################################

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
    nrows=3, ncols=2, figsize=(10, 10))
fig.suptitle(
    'Average Parameters')

ax1.bar(list(x1_B_data.keys()), list(x1_B_data.values()))
ax1.title.set_text(r'$<{B_{x1}}^2>$')
ax1.set_xticklabels(list(x1_B_data.keys()), rotation=75)

ax2.bar(list(x2_B_data.keys()), list(x2_B_data.values()))
ax2.title.set_text(r'$<{B_{x2}}^2>$')
ax2.set_xticklabels(list(x2_B_data.keys()), rotation=75)

ax3.bar(list(x1_Var_data.keys()), list(x1_Var_data.values()))
ax3.title.set_text(r'$<V_{x1}>$')
ax3.set_xticklabels(list(x1_Var_data.keys()), rotation=75)

ax4.bar(list(x2_Var_data.keys()), list(x2_Var_data.values()))
ax4.title.set_text(r'$<V_{x2}>$')
ax4.set_xticklabels(list(x2_Var_data.keys()), rotation=75)

ax5.bar(list(x1_MSE_data.keys()), list(x1_MSE_data.values()))
ax5.title.set_text(r'$<MSE_{x1}>$')
ax5.set_xticklabels(list(x1_MSE_data.keys()), rotation=75)

ax6.bar(list(x2_MSE_data.keys()), list(x2_MSE_data.values()))
ax6.title.set_text(r'$<MSE_{x2}>$')
ax6.set_xticklabels(list(x2_MSE_data.keys()), rotation=75)

plt.subplots_adjust(wspace=0.4,
                    hspace=1.0)
plt.savefig('pics/q4.png')
plt.close()


#    _________  ________  ___  ___       ___          ________             ________  ___  ___       ___     
#   |\___   ___\\_____  \|\  \|\  \     |\  \        |\   __  \           |\   ____\|\  \|\  \     |\  \    
#   \|___ \  \_|\|___/  /\ \  \ \  \    \ \  \       \ \  \|\  \  /\      \ \  \___|\ \  \ \  \    \ \  \   
#        \ \  \     /  / /\ \  \ \  \    \ \  \       \ \__     \/  \      \ \  \  __\ \  \ \  \    \ \  \  
#         \ \  \   /  /_/__\ \  \ \  \____\ \  \       \|_/  __     /|      \ \  \|\  \ \  \ \  \____\ \  \ 
#          \ \__\ |\________\ \__\ \_______\ \__\        /  /_|\   / /       \ \_______\ \__\ \_______\ \__\
#           \|__|  \|_______|\|__|\|_______|\|__|       /_______   \/         \|_______|\|__|\|_______|\|__|
#                                                       |_______|\__\                                       
#                                                               \|__|                                       
                                                                                                