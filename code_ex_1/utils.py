
########################################################################
#                      Ssp 1 - Programing Assigment                   
#   File Name: utils.py
#   Content: Helper function and methods fot spectral_estimation.py
#
#   Name: Yoav Ellinson                           Name: Chen Yaakov Sror
#   Id: 206036949                                 Id:   203531645
########################################################################

########################################################################
                        # Defines and Includes
########################################################################
import numpy as np

########################################################################
#                     Functions For Question 2
########################################################################

def get_Sx_bar(Sx, M=4096):
    Mc = len(Sx)
    Sx = np.array(Sx).reshape(Mc, int(M/2)+1)
    Sx_mean = np.sum(Sx, axis=0)/Mc
    return Sx_mean.flatten()

def get_bias(Sx_bar, Sx_true):
    if np.array(Sx_true).shape == np.array(Sx_bar).shape:
        return Sx_bar - Sx_true
    else:
        return None

def get_var(Sx, Sx_bar, M=4096):
    Mc = len(Sx)
    Sx = np.array(Sx).reshape(Mc, int(M/2)+1)
    var = np.zeros(Sx_bar.shape)
    for m in range(Mc):
        var += np.abs(Sx[m] - Sx_bar)**2
    return (var/Mc).flatten()

def get_MSE(var, bias):
    return var + bias**2

def total_bias(B):
    return np.sum(B**2)/len(B)

def total_var(var):
    return np.sum(var)/len(var)

def total_mse(mse):
    return np.sum(mse)/len(mse)

def get_all_stats(Sx, Sx_true, M=4096):
    Sx_bar = get_Sx_bar(Sx)
    B = get_bias(Sx_bar, Sx_true)
    var = get_var(Sx, Sx_bar)
    mse = get_MSE(var, B)
    bias_total = total_bias(B)
    mse_total = total_mse(mse)
    var_total = total_var(var)
    return Sx_bar, B, var, mse, bias_total, var_total, mse_total

########################################################################
#                        Helper Functions
########################################################################

def create_wl_mat(M=4096):
    w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]
    w = np.concatenate((-w_M[::-1], w_M[1:]))
    w_mat = np.repeat(w.reshape(w.shape[0], 1), w.shape[0], axis=1)
    l = np.zeros(w_mat.shape)
    l[0, :] = np.arange(-int(M/2), int(M/2)+1, 1)
    return np.dot(w_mat, l)

WL = create_wl_mat()  # matrix of ls and exp(-jwl)

########################################################################
#                        Periodegram Function
########################################################################

def calc_periodegram(x, M=4096):
    N = len(x)
    X = np.fft.fft(x, n=M)
    Sx_per = (1/N)*(np.abs(X)**2)[:int(M/2)+1]
    w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]
    return Sx_per, w_M

########################################################################
#                        Correlogram Function
########################################################################

def calc_correlogram(x, M=4096):
    N = len(x)
    x = np.pad(x, (0, int(M/2)+1-x.shape[0]), 'constant')
    w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]
    Rx_hat = np.correlate(x, x, 'full')/N
    Sx_cor = np.dot(np.exp(-1j*WL), Rx_hat).real[-2049:]
    return Sx_cor, w_M

########################################################################
#                        Functions For Question 3
########################################################################

########################################################################
#                         Blackman-Tukey Function
########################################################################

def calc_blackman_tukey(x, L=1, M=4096):
    N = len(x)
    x = np.pad(x, (0, int(M/2)+1-x.shape[0]), 'constant')
    w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]

    Rx_hat = np.correlate(x, x, 'full')/N
    l0 = np.argmax(Rx_hat)
    win = np.zeros(Rx_hat.shape)
    win[l0-L+1:l0+L] = np.ones((2*L-1))
    Rx_hat_win = np.multiply(Rx_hat, win)
    Sx_cor = np.dot(np.exp(-1j*WL), Rx_hat_win).real[-2049:]
    return Sx_cor, w_M

########################################################################
#                             Bartlet Function
########################################################################

def calc_bartlet(x, k, M=4096):
    N = len(x)
    if(not N % k):
        L = int(N/k)
        x_frames = [np.pad(x[(i*L):(i*L)+L], (0, M-L), 'constant')
                    for i in range(k)]
        Sx_k = np.array(
            [(np.abs(np.fft.fft(x_frames[i]))**2)/L for i in range(k)])
        Sx_b = np.array((1/k)*(np.sum(Sx_k, axis=0)))[:int(M/2)+1]
        w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]
        return Sx_b, w_M
    else:
        print('not a valid value for k')
        return None, None

########################################################################
#                             Welch Function
########################################################################

def calc_welch(x, k, D, L, M=4096):
    N = len(x)
    x_frames = np.array([np.pad(x[i:i+L], (0, M-L), 'constant')
                        for i in np.multiply(list(range(k)), L-D)])
    Sx_k = np.array([(np.abs(np.fft.fft(x_frames[i]))**2)/L for i in range(k)])
    Sx_b = np.array((1/k)*(np.sum(Sx_k, axis=0)))[:int(M/2)+1]
    w_M = np.linspace(0, 2*np.pi, M)[:int(M/2)+1]
    return Sx_b, w_M


#    _________  ________  ___  ___       ___          ________             ________  ___  ___       ___     
#   |\___   ___\\_____  \|\  \|\  \     |\  \        |\   __  \           |\   ____\|\  \|\  \     |\  \    
#   \|___ \  \_|\|___/  /\ \  \ \  \    \ \  \       \ \  \|\  \  /\      \ \  \___|\ \  \ \  \    \ \  \   
#        \ \  \     /  / /\ \  \ \  \    \ \  \       \ \__     \/  \      \ \  \  __\ \  \ \  \    \ \  \  
#         \ \  \   /  /_/__\ \  \ \  \____\ \  \       \|_/  __     /|      \ \  \|\  \ \  \ \  \____\ \  \ 
#          \ \__\ |\________\ \__\ \_______\ \__\        /  /_|\   / /       \ \_______\ \__\ \_______\ \__\
#           \|__|  \|_______|\|__|\|_______|\|__|       /_______   \/         \|_______|\|__|\|_______|\|__|
#                                                       |_______|\__\                                       
#                                                               \|__|                                       
                                                                                                