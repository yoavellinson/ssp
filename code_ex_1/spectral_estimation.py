import numpy as np
import matplotlib.pyplot as plt
from utils import Rxx_hat_b
from tqdm import tqdm

#Q1
def Sxx1(w):
    return 1 +(9/13)*np.cos(w) -(4/13)*np.cos(2*w)
def Sxx2(w):
    return 0.86/(np.abs(1-0.7*np.exp(-1j*w))**2)
n=2049
w = np.linspace(0,np.pi,num=n)

# Q2.a.1

sigma_w1 = np.sqrt(1/26)
sigma_w2 = np.sqrt(0.86)

n1 =1024
n2 = 2048


def gen_signals(sigma_w1=np.sqrt(1/26),n1=1024,sigma_w2=np.sqrt(0.86),n2=2048):
    w1 = np.random.normal(0,sigma_w1,n1)
    x1 = np.array([w1[n]-3*w1[n-1]-4*w1[n-2] for n in range(n1)])

    w2 = np.random.normal(0,sigma_w2,n2)

    def gen_x2(x2,n):
        return 0.7*x2[n-1] + w2[n]
    x2_tmp = np.zeros(n2)
    for n in range(1,n2):
        x2_tmp[n] = gen_x2(x2_tmp,n)

    x2 = x2_tmp[-1024:]
    return x1,x2


def calc_periodegram(x,M=4096):
    x = np.pad(x,(0,M-x.shape[0]),'constant')
    X = np.fft.fft(x,n=M) 
    Sx_per = (1/len(x))*(np.abs(X)**2)[:int(M/2)+1]
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    return Sx_per,w_M


def calc_correlogram(x,M=4096):
    x = np.pad(x,(0,int(M/2)+1-x.shape[0]),'constant')
    N=len(x)
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    Rx_hat = np.correlate(x,x,'full')/N
    w = np.concatenate((-w_M[::-1],w_M[1:]))
    w_mat = np.repeat(w.reshape(w.shape[0],1),w.shape[0],axis=1)
    l = np.zeros(w_mat.shape)
    l[0,:] = np.arange(-int(M/2),int(M/2)+1,1)
    wl = np.dot(w_mat,l)
    Sx_cor = np.dot(np.exp(-1j*wl),Rx_hat).real[-2049:]
    return Sx_cor,w_M

def calc_blackman_tukey(x,L=1,M=4096):
    N=len(x)
    x = np.pad(x,(0,int(M/2)+1-x.shape[0]),'constant')
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    Rx_hat = np.correlate(x,x,'full')/N
    l0 = np.argmax(Rx_hat)
    win = np.zeros(Rx_hat.shape)
    win[l0-L+1:l0+L] = np.ones((2*L-1))
    Rx_hat_win = np.multiply(Rx_hat,win)
    w = np.concatenate((-w_M[::-1],w_M[1:]))
    w_mat = np.repeat(w.reshape(w.shape[0],1),w.shape[0],axis=1)
    l = np.zeros(w_mat.shape)
    l[0,:] = np.arange(-int(M/2),int(M/2)+1,1)
    wl = np.dot(w_mat,l)
    Sx_cor = np.dot(np.exp(-1j*wl),Rx_hat_win).real[-2049:]
    return Sx_cor,w_M

def calc_bartlet(x,k,M=4096):
    N = len(x)
    if(not N%k):
        L = int(N/k)
        x_frames = [np.pad(x[(i*L):(i*L)+L],(0,M-L),'constant') for i in range(k)]
        Sx_k = np.array([(np.abs(np.fft.fft(x_frames[i]))**2)/L for i in range(k)])
        Sx_b = np.array(1/k)*(np.sum(Sx_k,axis=0))[:int(M/2)+1]
        w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
        return Sx_b,w_M
    else:
        print('not a valid value for k')
        return None,None

def calc_welch(x,k,D,L,M=4096):
    N = len(x)
    x_frames = np.array([np.pad(x[i:i+L],(0,M-L),'constant') for i in np.multiply(list(range(k)),L-D)])
    Sx_k = np.array([(np.abs(np.fft.fft(x_frames[i]))**2)/L for i in range(k)])
    Sx_b = np.array((1/k)*(np.sum(Sx_k,axis=0)))[:int(M/2)+1]
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    return Sx_b,w_M
    

Mc = 10
Sx1_per = np.empty((Mc,2049))
Sx1_bartlet_64 = np.empty((Mc,2049))
Sx1_bartlet_16 = np.empty((Mc,2049))
Sx1_welch_64 = np.empty((Mc,2049))
Sx1_welch_16 = np.empty((Mc,2049))
Sx1_bt_4 = np.empty((Mc,2049))
Sx1_bt_2 = np.empty((Mc,2049))

Sx2_per = np.empty((Mc,2049))
Sx2_bartlet_64 = np.empty((Mc,2049))
Sx2_bartlet_16 = np.empty((Mc,2049))
Sx2_welch_64 = np.empty((Mc,2049))
Sx2_welch_16 = np.empty((Mc,2049))
Sx2_bt_4 = np.empty((Mc,2049))
Sx2_bt_2 = np.empty((Mc,2049))

for i in tqdm(range(Mc)):
    x1,x2=gen_signals()
    Sx1_per[i],_= calc_periodegram(x1) 
    Sx1_bartlet_64[i],_ = calc_bartlet(x1,64) 
    Sx1_bartlet_16[i],_ =calc_bartlet(x1,16) 
    Sx1_welch_64[i],_ = calc_welch(x1,k=61,L=64,D=48) 
    Sx1_welch_16[i],_ = calc_welch(x1,k=253,L=16,D=12) 
    Sx1_bt_4[i],_ = calc_blackman_tukey(x1,L=4)
    Sx1_bt_2[i],_ = calc_blackman_tukey(x1,L=2)

    Sx2_per[i],_= calc_periodegram(x2) 
    Sx2_bartlet_64[i],_ = calc_bartlet(x2,64) 
    Sx2_bartlet_16[i],_ = calc_bartlet(x2,16) 
    Sx2_welch_64[i],_ = calc_welch(x2,k=61,L=64,D=48)
    Sx2_welch_16[i],_ = calc_welch(x2,k=253,L=16,D=12)
    Sx2_bt_4[i],_ =  calc_blackman_tukey(x2,L=4)
    Sx2_bt_2[i],w_M = calc_blackman_tukey(x2,L=2)

def get_avg(Sx,M=4096):
    N = len(Sx)
    Sx = np.array(Sx).reshape(N,int(M/2)+1)
    Sx_mean = np.sum(Sx,axis=0)/N
    return Sx_mean.flatten()

Sx1_per_Mc = get_avg(Sx1_per)
Sx1_bartlet_64_Mc = get_avg(Sx1_bartlet_64)
Sx1_bartlet_16_Mc = get_avg(Sx1_bartlet_16)
Sx1_welch_64_Mc = get_avg(Sx1_welch_64)
Sx1_welch_16_Mc = get_avg(Sx1_welch_16)
Sx1_bt_4_Mc = get_avg(Sx1_bt_4)
Sx1_bt_2_Mc = get_avg(Sx1_bt_2)

Sx2_per_Mc = get_avg(Sx2_per)
Sx2_bartlet_64_Mc = get_avg(Sx2_bartlet_64)
Sx2_bartlet_16_Mc = get_avg(Sx2_bartlet_16)
Sx2_welch_64_Mc = get_avg(Sx2_welch_64)
Sx2_welch_16_Mc = get_avg(Sx2_welch_16)
Sx2_bt_4_Mc = get_avg(Sx2_bt_4)
Sx2_bt_2_Mc = get_avg(Sx2_bt_2)

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,10))
fig.suptitle(r'$\bar{S}_{x1}$ and $\bar{S}_{x2}$ Estimations for Mc'+f'={Mc}')
ax1.title.set_text(r'$\bar{S}_{x1}$')
ax1.set_xlabel(r'$\omega$')
ax2.title.set_text(r'$\bar{S}_{x2}$')
ax2.set_xlabel(r'$\omega$')
ax1.plot(w_M,Sx1_per_Mc,label = 'Sx1 Periodegram ($e^{jw} $)')
ax1.plot(w_M,Sx1_bt_2_Mc,label = 'Sx1 BT L=2 ($e^{jw} $)')
ax1.plot(w_M,Sx1_bt_4_Mc,label = 'Sx1 BT L=4 ($e^{jw} $)')
ax1.plot(w_M,Sx1_bartlet_16_Mc,label = 'Sx1 Bartlet L=16 ($e^{jw} $)')
ax1.plot(w_M,Sx1_bartlet_64_Mc,label = 'Sx1 Bartlet L=64 ($e^{jw} $)')
ax1.plot(w_M,Sx1_welch_16_Mc,label = 'Sx1 Welch L=16 ($e^{jw} $)')
ax1.plot(w_M,Sx1_welch_64_Mc,label = 'Sx1 Welch L=64 ($e^{jw} $)')
ax2.plot(w_M,Sx2_per_Mc,label = 'Sx2 Periodegram ($e^{jw} $)')
ax1.plot(w_M,Sxx1(w_M),linewidth=1.5,label = 'Sx1 True($e^{jw} $)')
ax2.plot(w_M,Sx2_bt_2_Mc,label = 'Sx2 BT L=2 ($e^{jw} $)')
ax2.plot(w_M,Sx2_bt_4_Mc,label = 'Sx2 BT L=4 ($e^{jw} $)')
ax2.plot(w_M,Sx2_bartlet_16_Mc,label = 'Sx2 Bartlet L=16 ($e^{jw} $)')
ax2.plot(w_M,Sx2_bartlet_64_Mc,label = 'Sx2 Bartlet L=64 ($e^{jw} $)')
ax2.plot(w_M,Sx2_welch_16_Mc,label = 'Sx2 Welch L=16 ($e^{jw} $)')
ax2.plot(w_M,Sx2_welch_64_Mc,label = 'Sx2 Welch L=64 ($e^{jw} $)')
ax2.plot(w_M,Sxx2(w_M),linewidth=1.5,label = 'Sx2 True ($e^{jw} $)')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.show()
