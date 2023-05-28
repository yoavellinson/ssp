import numpy as np
import matplotlib.pyplot as plt



def plot_roots(pol_coeff):
    #unit circle
    t = np.linspace(0,2*np.pi,101)
    plt.plot(np.cos(t),np.sin(t))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.grid()

    #polynomial roots
    r = np.roots(pol_coeff)
    r_real,r_imag = [e.real for e in r],[e.imag for e in r]
    plt.scatter(r_real,r_imag)
    plt.scatter(0,0,c='black')
    plt.show()

plot_roots([1,2,3,4,5,4,3,2,1])