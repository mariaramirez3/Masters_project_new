import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg for Linux
import matplotlib.pyplot as plt

def get_dat(file_name):

    df = pd.read_fwf(file_name +'.dat', delimiter=' ', comment='#', header=None, skiprows=4)  # Specify your delimiter here
    Rx = df[0].to_numpy()
    Ry = df[1].to_numpy()
    # Rz = df[2].to_numpy() not needed

    tr = df[5].to_numpy()
    ti = df[6].to_numpy()

    return(Rx, Ry, tr, ti)

rx1,ry1,tr1,ti1 = get_dat("1NN_hr")

#2d hamiltonian
def diag_h(kx,ky, rx, ry, tr, ti):
    len_ks = len(kx)
    len_ts = len(tr)

    egs = []

    for j in range(len_ks):

        egs.append(sum((tr[i]+ti[i]) * np.exp(kx[j]*rx[i] + ky[j]*ry[i]) for i in range(len_ts)))
    return(egs)

#values of k in 1st Brillouin Zone
kx = np.arange(-np.pi, np.pi, 0.02)
ky = np.arange(-np.pi, np.pi, 0.02)

Eg1 = diag_h(kx,ky,rx1,ry1,tr1,ti1)

k_vecs = np.array([np.sqrt(kx[i]**2 +ky[i]**2) for i in range(len(kx))])

fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.plot(k_vecs, Eg1)
plt.show()




