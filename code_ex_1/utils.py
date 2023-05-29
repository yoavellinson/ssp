import numpy as np


def Rxx_hat_b(x,two_sided = False): # creates the auto-corollation series of x, if two sided it creates it for the negative l values as well
    sum_n =np.array([np.dot(x[l:l+len(x)-1-abs(l)],x[0:len(x)-1-abs(l)].T) for l in range(len(x))])
    if two_sided:
        tmp = np.zeros(len(x)*2 -1)
        tmp[len(x)-1:] = sum_n
        tmp[:len(x)] = sum_n[::-1]
        return (1/len(x))*tmp
    else:
        return (1/len(x))*sum_n
    



def calc_periodegram_correlogram(x,M=4096): #returns Sx_periodogram,Sx_correlogram, w - all the correct omegas 
    #old function
    #periodogram
    x = np.pad(x,(0,int(M/2)+1-x.shape[0]),'constant')
    X = np.fft.fft(x,n=M) 
    Sx_per = (1/M)*(np.abs(X)**2)[:int(M/2)+1]
    w_M = np.linspace(0,2*np.pi,M)[:int(M/2)+1]
    #correlogram
    # Rx_hat =Rxx_hat_b(x,two_sided=True) #with a loop - not accurate enough
    Rx_hat = np.correlate(x,x,'full')/M
    l0 = np.argmax(Rx_hat)
    # Sx_cor = [np.sum([Rx_hat[l]*np.exp(-1j*w*(l0-l)) for l in range(Rx_hat.shape[0])]) for w in w_M] #with a loop
    w = np.concatenate((-w_M[::-1],w_M[1:])) #with matrix multipication
    w_mat = np.repeat(w.reshape(w.shape[0],1),w.shape[0],axis=1)
    l = np.zeros(w_mat.shape)
    l[0,:] = np.arange(-int(M/2),int(M/2)+1,1)
    wl = np.dot(w_mat,l)
    Sx_cor = np.dot(np.exp(-1j*wl),Rx_hat).real[-2049:]
    return Sx_per,Sx_cor,w_M

