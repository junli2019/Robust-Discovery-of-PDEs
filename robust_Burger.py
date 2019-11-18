# Robust PDE-FIND for Burger's Equation

#% pylab inline

import pylab
pylab.rcParams['figure.figsize'] = (12, 8)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from robust_PDE_FIND import build_linear_system, TrainSTRidge, compute_err, print_pde, RobustPCA, Robust_LRSTR
import scipy.io as scio
from scipy.sparse import random
import itertools
import matplotlib.pyplot as plt


data = scio.loadmat('burgers.mat')
u = np.real(data['usol'])
x = np.real(data['x'][0])
t = np.real(data['t'][:,0])
dt = t[1]-t[0]
dx = x[2]-x[1]


#X, T = np.meshgrid(x, t)
#fig1 = plt.figure()
#ax = fig1.gca(projection='3d')
#surf = ax.plot_surface(X, T, u.T, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
#    linewidth=0, antialiased=False)
#plt.title('Burgers Equation', fontsize = 20)
#plt.xlabel('x', fontsize = 16)
#plt.ylabel('t', fontsize = 16)
#plt.show()



#################################################################################
############################ No noise ###########################################
Ut, R, rhs_des = build_linear_system(u, dt, dx, D=3, P=3, time_diff = 'FD', space_diff = 'FD')
['1'] + rhs_des[1:]

# Solve with STR
w = TrainSTRidge(R,Ut,10**-5,1)
print("PDE derived using STR for clear data U")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  1.000987)*100, (.1 - 0.100220)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")




# Solve with RPCA + STR
nx,nt = 256, 101
Z, E1 = RobustPCA(u, lam_2 = 0.3)
Ut, R, rhs_des = build_linear_system(Z, dt, dx, D=3, P=3, time_diff = 'FD', space_diff = 'FD')

w = TrainSTRidge(R,Ut,10**-5,1)
print("PDE derived using RPCA+LRSTR for noise data U")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  1.000582)*100, (.1 - 0.100189)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")

# Solve with RPCA+LRSTR
Ut1 = np.reshape(Ut,(nx,nt))
w, X, E2=Robust_LRSTR(R, Ut1, rhs_des, lam_1 = 1e-5, lam_3 = 0.3, lam_4 = 1e-4, d_tol = 1)
print("PDE derived using RPCA+LRSTR for clear data U")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  0.999367)*100, (.1 - 0.100089)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")

#################################################################################
############################ E1 noise ###########################################
np.random.seed(0)
noise = np.random.randn(u.shape[0],u.shape[1])
# Gaussian noise: un = u + 0.05*np.std(u)*noise 
# sparse noise
mask= np.random.uniform(0,1,(u.shape[0],u.shape[1]))
sparsemask=np.where(mask>0.9,1,0)
un = u + 1*np.std(u)*noise*sparsemask

scio.savemat('burger_Un.mat', {'un': un})

Utn, Rn, rhs_des = build_linear_system(un, dt, dx, D=3, P=3, time_diff = 'poly',
                                       deg_x = 4, deg_t = 4, 
                                       width_x = 10, width_t = 10)
['1'] + rhs_des[1:]
# Solve with STR
w = TrainSTRidge(Rn,Utn,10**-5,1)
print("PDE derived using STR for noise data U")
print_pde(w, rhs_des)


# Solve with RPCA + STR
nx,nt = 236, 81
Z, E1 = RobustPCA(un, lam_2 = 0.1)

scio.savemat('burger_Z.mat', {'Z': Z})
scio.savemat('burger_E1.mat', {'E1': E1})

ZUtn, ZRn, rhs_des = build_linear_system(Z, dt, dx, D=3, P=3, time_diff = 'poly',
                                       deg_x = 4, deg_t = 4, 
                                       width_x = 10, width_t = 10)

scio.savemat('burger_ZUtn.mat', {'ZUtn': ZUtn})
scio.savemat('burger_ZRn.mat', {'ZRn': ZRn})

w = TrainSTRidge(ZRn,ZUtn,10**-5,1)
print("PDE derived using RPCA+RLRSTR for noise data U")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  1.010024)*100, (.1 - 0.105399)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")

# Solve with RPCA+LRSTR
Utn1 = np.reshape(ZUtn,(nx,nt))
w, X, E2=Robust_LRSTR(ZRn, Utn1, rhs_des, lam_1 = 1e-5, lam_3 = 0.15, lam_4 = 1, d_tol = 1)
scio.savemat('burger_X.mat', {'X': X})
scio.savemat('burger_E2.mat', {'E2': E2})

print("PDE derived using RPCA+RLRSTR for noise data U")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  0.977689)*100, (.1 - .1003118)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")


#################################################################################
############################ E1 + E2 noises #####################################
np.random.seed(0)
noise = np.random.randn(u.shape[0],u.shape[1])
# Gaussian noise: un = u + 0.05*np.std(u)*noise 
# sparse noise
mask= np.random.uniform(0,1,(u.shape[0],u.shape[1]))
sparsemask=np.where(mask>0.9,1,0)
un = u + 1*np.std(u)*noise*sparsemask

Utn, Rn, rhs_des = build_linear_system(un, dt, dx, D=3, P=3, time_diff = 'poly',
                                       deg_x = 4, deg_t = 4, 
                                       width_x = 10, width_t = 10)

# sparse noise for Ut
nx,nt = 236, 81
Utn = np.reshape(Utn,(nx,nt))
noise = np.random.randn(nx,nt)
mask= np.random.uniform(0,1,(nx,nt))
sparsemask=np.where(mask>0.9,1,0)
Utn = Utn + 2*np.std(Utn)*noise*sparsemask
Utn = np.reshape(Utn,(nx*nt,1))

# Solve with STR
w = TrainSTRidge(Rn,Utn,10**-5,1)
print("PDE derived using STR for both noise data U and Ut")
print_pde(w, rhs_des)

# Solve with RPCA + STR
nx,nt = 236, 81
Z, E1 = RobustPCA(un, lam_2 = 0.1)
ZUtn, ZRn, rhs_des = build_linear_system(Z, dt, dx, D=3, P=3, time_diff = 'poly',
                                       deg_x = 4, deg_t = 4, 
                                       width_x = 10, width_t = 10)
Utn = np.reshape(ZUtn,(nx,nt))
Utnn = Utn + 2*np.std(Utn)*noise*sparsemask

scio.savemat('burger_Utnn.mat', {'Utnn': Utnn})
Utn = np.reshape(Utnn,(nx*nt,1))

w = TrainSTRidge(ZRn,Utn,10**-5,1)
print("PDE derived using RPCA+STR for both noise data U and Ut")
print_pde(w, rhs_des)

# Solve with RPCA + LRSTR
w, X, E2=Robust_LRSTR(ZRn, Utnn, rhs_des, lam_1 = 1e-5, lam_3 = 0.15, lam_4 = 1, d_tol = 1)
print("PDE derived using RPCA+RLRSTRidge for both noise data U and Ut")
print_pde(w, rhs_des)
err = np.abs(np.array([(1 -  0.956320)*100, (.1 - 0.101799)*100/0.1]))
print("Mean error:", np.mean(err), '%', "Std:", np.std(err), '%')
print(" ")

