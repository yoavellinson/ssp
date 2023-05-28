import numpy as np
import matplotlib.pyplot as plt


#Q1


def Sxx1(w):
    return 1 +(9/13)*np.cos(w) -(4/13)*np.cos(2*w)
def Sxx2(w):
    return 0.86/(np.abs(1-0.7*np.exp(-1j*w))**2)

n=2049
w = np.linspace(0,np.pi,num=n)

plt.plot(w,Sxx1(w),label = 'Sxx1(exp(jw))')
plt.plot(w,Sxx2(w),label = 'Sxx2(exp(jw))')
plt.legend()
# plt.savefig('pics/q1.png') #for debuging
plt.close()

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
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
ax1.plot(x1,label = 'x1[n]')
ax1.set_xlabel('n')
ax1.set_ylabel('x1[n]')
ax1.legend()
ax2.plot(x2,label = 'x2[n]')
ax2.legend()
ax2.set_xlabel('n')
ax2.set_ylabel('x2[n]')
plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
# plt.savefig('pics/q2a_1.png') #for debuging
plt.close()


#Q2.a.2 Ask yuval

#Q2b
#since x1 has 1024 samples and we want 4096 samples we need to pad with zeros - numpy does it automaticly when setting n to be altger the the shape on x.
M = 4096
X1 = np.fft.fft(x1,n=M)

Sx1_per = (1/4096)*(np.abs(X1)**2)[:int(M/2)+1]
w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
plt.plot(w_M,[Sx1_per[w] for w in range(len(w_M))],label = 'Sx1_periodegram(exp(jw))')
plt.plot(w,Sxx1(w),label = 'Sxx1(exp(jw))')
plt.legend()
plt.close()

# Q2c

def R_hat_b(x):
    tmp = np.zeros(len(x1)*2 -1)
    sum_n =np.array([np.dot(x[l:l+len(x)-1-abs(l)],x[0:len(x)-1-abs(l)].T) for l in range(len(x))])
    tmp[len(x)-1:] = sum_n
    tmp[:len(x)] = sum_n[::-1]
    return (1/len(x))*tmp
N= len(x1)

data_mat =np.zeros((N+N-1,N))
for l in range(N):
    data_mat[l:l+N,l] = x1
Rx1_hat = 1/n * np.dot(data_mat.T,data_mat)[0]
tmp = np.zeros(N*2 -1)
tmp[N-1:] = Rx1_hat
tmp[:N] = Rx1_hat[::-1]
Rx1_hat = tmp #ASK YUBVAL ABOUT THE NEGATIVE VALUES
# Rx1_hat =R_hat_b(x1)
Sx1_cor = np.fft.fft(Rx1_hat,M)[:int(M/2)+1]# not good! change sum from -(N-1) to N-1
plt.plot(Rx1_hat,label = 'Rxx1(l)') 
plt.legend()
plt.show()

plt.plot(w_M,[Sx1_cor[w] for w in range(len(w_M))],label = 'Sx1_corelogram(exp(jw))')
plt.plot(w,Sxx1(w),label = 'Sxx1(exp(jw))')
plt.legend()
plt.show()

