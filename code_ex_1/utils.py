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
    


