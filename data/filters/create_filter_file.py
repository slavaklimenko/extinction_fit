import numpy as np
import matplotlib.pyplot as plt
lmin,lmax = 4.95,5.74
delta = 0.1
res = 3500
x = np.linspace(lmin-delta,lmax+delta,int((lmax-lmin+2*-delta)/(lmax+lmin)*2*res))
x*=1e4
y = np.zeros_like(x)
y[(x>=lmin*1e4)*(x<=lmax*1e4)] = 1
s = np.trapz(y,x)
d = np.zeros((len(x),2))
d[:,0] = x
d[:,1] = y/s
np.savetxt('./JWST_MIRI.1A.dat',d)

plt.plot(x,y/s)
plt.show()