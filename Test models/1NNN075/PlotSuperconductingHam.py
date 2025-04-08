import Hamiltonian
import matplotlib.pyplot as plt
import numpy as np
import os
import re
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


import matplotlib.pyplot as plt
font = {'family': 'arial',
        'weight': 'bold',
        'size': 12}

plt.rc('font', **font)
import numpy as np




def main(): 
    
    #fix_chemical_potential(t3)
    #exit(0)
    Ymin = -0.5
    Ymax = 0.5
    #Take tight binding model 
    counter = 0
      #params = ["0.020","0.0638"]
    #params = ["6.00","0.020","0.0638"]
    #params = ["1.00","-0.050","-0.1850"]

    params = ["3.00","0.020","0.0638"]
    #params = ["2.00","0.000","0.0150"]

    #params = ["2.50","0.005","0.0350"]
    #params = ["2.50","0.006","0.0390"]
    #params = ["2.50","0.007","0.0430"]
    #params = ["2.50","0.008","0.0469"]
    #params = ["6.00","0.020","0.0638"]
    U = params[0]
    t3 = params[1]
    mu = params[2]
    ff = "4.00"
    
    model_normal = f"SRH_t{t3}_mu{mu}.dat"
    SCgap_file = f"gap_file_t{t3}_mu{mu}_U{U}_ff{ff}.txt"
    model_SCgap = f"SRH_t{t3}_mu{mu}_U{U}_SC.dat"

    orange = "#f5b41d" # a nice orange
    purple = "#a349a4" # a nice purple

    Ham = Hamiltonian.tbHamiltonian(model_SCgap,FermiEnergy = 0.0,no_kgrid=True,superconducting=True)

    GapAngle(Ham)
    exit(0)

    Ham.DefineOrbitals(["z2"],[purple]) #DEFINE ORBITALS HERE
    Ham.Load_kgrid(256,256,1)    



    Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

    Gamma = [0.0,0.0,0.0] #in units of 2pi*b1,2pi*b2, 2pi*b3 where b1,b2,b3 are the reciprocal lattice vectors.
    Y = [0.0,0.5,0.0]
    X = [0.5,0.0,0.0] #for a square lattice with a=b=c=1 this would be the (pi,pi) point. 
    M = [0.5,0.5,0]
    Mp =[-0.5, 0.5,0.0]
    Z = [0.0,0.0,0.5]
    R = [0.5,0.0,0.5]
    A = [0.5,0.5,0.5]
    kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
    klabels = [r"$\Gamma$","X","Mx",r"$\Gamma$"] #Write your labels as well. 
  
    #bands_x,bands_y = Hamplot.BandStructure(kpath=kpath,klabels=klabels,YMIN=Ymin,YMAX=Ymax,linewidth=2,nk=99,Use_KPath_Distance=True)
    #plt.show()
    Delta = Hamiltonian.GetSCGap(Ham.GetHoppings(),Ham.nbands,Ham.lattice_vectors,2.845,0.0,0,SpinPolarised=False)
    print(Delta)
    Delta = Hamiltonian.GetSCGap(Ham.GetHoppings(),Ham.nbands,Ham.lattice_vectors,3.166,1.18,0,SpinPolarised=False)
    print(Delta)
    #exit(0)
    Hamplot.plotGapAtEF(N_BrillouinZone=2)
    plt.show()
    exit(0)
    #Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,N_BrillouinZone=2,linewidth=2,kz=0)
    #plt.show()
    #Hamplot.ax_FS_orbColour.set_title(f"t3 = {np.round(t,3)}eV")

    """
        Gamma = [0.0,0.0,0.0] #in units of 2pi*b1,2pi*b2, 2pi*b3 where b1,b2,b3 are the reciprocal lattice vectors.
        Y = [0.0,0.5,0.0]
        X = [0.5,0.0,0.0] #for a square lattice with a=b=c=1 this would be the (pi,pi) point. 
        M = [0.5,0.5,0]
        Mp =[-0.5, 0.5,0.0]
        Z = [0.0,0.0,0.5]
        R = [0.5,0.0,0.5]
        A = [0.5,0.5,0.5]
        kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
        klabels = [r"$\Gamma$","X","Mx",r"$\Gamma$"] #Write your labels as well. 
    
        bands_x,bands_y = Hamplot.BandStructure(kpath=kpath,klabels=klabels,YMIN=Ymin,YMAX=Ymax,linewidth=2,nk=99,Use_KPath_Distance=True)
        plt.clf()
        plt.close()
        dos_x,dos_y=Hamplot.PlotDOS(E_start=-0.5,E_end=0.5,Nsteps=401,Gamma=0.001)
        plt.clf()
        plt.close()
        #add axes to a single 2 panel figure
        fig, axs = plt.subplots(1,2,figsize=(7,6))
        axs[0].plot(bands_x,bands_y,linewidth=3,c="#508484")
        #axs[0].set_aspect(50)
        axs[0].set_xlim(bands_x[0],bands_x[-1])
        axs[0].set_ylim(Ymin,Ymax)
        #add horizontal line at zero
        axs[0].hlines(0,0,bands_x[-1],color="Grey",lw=1,linestyles='dashed')
        
        axs[0].set_ylabel("Energy (eV)")
        xsize=len(bands_x)
        axs[0].set_xticks([bands_x[0],bands_x[int(xsize/3.0)],bands_x[int(2.0*xsize/3.0)],bands_x[-1]])
        axs[0].set_xticklabels([r"$\Gamma$","X","M",r"$\Gamma$"])

        axs[0].vlines(bands_x[int(xsize/3.0)],Ymin,Ymax,color="Grey",lw=1,linestyles='dashed')
        axs[0].vlines(bands_x[int(2.0*xsize/3.0)],Ymin,Ymax,color="Grey",lw=1,linestyles='dashed')
        axs[0].set_xlabel("k")
        axs[0].set_title("Band Structure")
        axs[0].set_title(f"t3 = {np.round(t,3)}eV")

        DOS_lim = 10
        axs[1].plot(dos_y[0],dos_x,linewidth=3,c="#508484")
        #axs[1].set_aspect(50)
        axs[1].set_xlim(0,DOS_lim)
        axs[1].hlines(0,0,DOS_lim,color="Grey",lw=1,linestyles='dashed')
        axs[1].set_ylim(Ymin,Ymax)
        #axs[1].set_ylabel("Energy (eV)")
        axs[1].set_xlabel("DOS")

        plt.savefig(f"SRH_fixed_{counter}.png")
        counter+=1

        #plt.show()
        """
    exit(0)
    #FS

    #Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,N_BrillouinZone=2,linewidth=2,kz=0)
    #plt.savefig("FermiSurface.png")

    #DOS
    #plt.savefig("DOS.png")


if __name__ == "__main__":
    main()