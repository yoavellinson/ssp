import numpy as np
import matplotlib.pyplot as plt
from utils import Rxx_hat_b

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

w1 = np.random.normal(0,sigma_w1,n1)
x1 = np.array([w1[n]-3*w1[n-1]-4*w1[n-2] for n in range(n1)])

w2 = np.random.normal(0,sigma_w2,n2)
def gen_x2(x2,n):
    return 0.7*x2[n-1] + w2[n]

x2_tmp = np.zeros(n2)
for n in range(1,n2):
    x2_tmp[n] = gen_x2(x2_tmp,n)

x2 = x2_tmp[-1024:]



#Q2b+c

def calc_spectrum(x,M=4096): #returns Sx_periodogram,Sx_correlogram, w - all the correct omegas
    #periodogram
    x = np.pad(x,(0,int(M/2)+1-x.shape[0]),'constant')
    X = np.fft.fft(x,n=M) 
    Sx_per = (1/M)*(np.abs(X)**2)[:int(M/2)+1]
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    #correlogram
    Rx_hat =Rxx_hat_b(x,two_sided=True)
    l0 = np.argmax(Rx_hat)
    # Sx_cor = [np.sum([Rx_hat[l]*np.exp(-1j*w*(l0-l)) for l in range(Rx_hat.shape[0])]) for w in w_M] #with a loop
    w = np.concatenate((-w_M[::-1],w_M[1:])) #with matrix multipication
    w_mat = np.repeat(w.reshape(w.shape[0],1),w.shape[0],axis=1)
    l = np.zeros(w_mat.shape)
    l[0,:] = np.arange(-int(M/2),int(M/2)+1,1)
    wl = np.dot(w_mat,l)
    Sx_cor = np.dot(np.exp(-1j*wl),Rx_hat).real[-2049:]
    return Sx_per,Sx_cor,w_M


Sx1_per, Sx1_cor,w_1 = calc_spectrum(x1)
Sx2_per, Sx2_cor,w_2= calc_spectrum(x2)

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,13))
ax1.plot(w_1,[Sx1_per[w] for w in range(len(w_1))],label = 'Sx1_periodogram($e^{jw} $)')
ax1.plot(w_1,[Sx1_cor[w] for w in range(len(w_1))],label = 'Sx1_correlogram($e^{jw} $)')
ax1.plot(w_1,Sxx1(w_1),label = 'Sxx1($e^{jw} $)')
ax1.legend()
ax1.title.set_text('$X_1$ Spectrum')

ax2.plot(w_2,[Sx2_per[w] for w in range(len(w_2))],label = 'Sx2_periodogram($e^{jw} $)')
ax2.plot(w_2,[Sx2_cor[w] for w in range(len(w_2))],label = 'Sx2_correlogram($e^{jw} $)')
ax2.plot(w_2,Sxx2(w_2),label = 'Sxx2($e^{jw} $)')
ax2.legend()
ax2.title.set_text('$X_2$ Spectrum')

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.show()
# plt.savefig('pics/q2.png')
