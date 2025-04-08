# to load and plot cqpi .idl files calculated using CalcQPI:
import Hamiltonian
import matplotlib.pyplot as plt
import numpy as np
import os

#os.system("pip3 install tqdm")
#exit(0)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
###############################
###############################
##BASIC USAGE
# Load the tight binding Hamiltonian (the .dat file) using Hamiltonian.tbHamiltonian('Sr214M_hr.dat') if the FermiEnergy is not 0 then you can specify it here. 
# Load the eigenvalues and eigenvectors over the first Brillouin zone using Ham.Load_kgrid(nkx,nkz,nkz). If you make this number too big you'll run out of memory. start small and increase the size if required. 
# Create a Hamiltonian plot object using Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
# Plot your desired property using the methods in Hamplot. 

###############################

###############################
#----------------------------------------------------------
##Example 1 - plotting band structure
#----------------------------------------------------------
#For the bandstructure you dont need to load the kgrid. You must specify the high symmetry path you want to follow (kpath) and how many points you want on each line in that path (nk). 
#YMIN and YMAX are the limits of the y axis.
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Gamma = [0.0,0.0,0.0] #in units of 2pi*b1,2pi*b2, 2pi*b3 where b1,b2,b3 are the reciprocal lattice vectors.
X = [0.5,0.5,0] #for a square lattice with a=b=c=1 this would be the (pi,pi) point. 
M = [0.5,0,0]
kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
klabels = [r"$\Gamma$","X","M",r"$\Gamma$"] #Write your labels as well. 
Hamplot.BandStructure(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100,color="#680C07")
plt.show()

#exit(0)

###############################
#----------------------------------------------------------
##Example 2 - plotting orbitally resolved band structure
#----------------------------------------------------------
#Extracting orbital information can be very important. you need to first specify the orbitals present in the model using tbHamiltonian.DefineOrbitals([Orbital labels], [Orbital_Colours])
#Then call BandStructure_orbital_resolved() which requires the same arguemnts as BandStructure(), but plots the maximum orbital character of the band as you have specified above.  
#You do not have to define every orbital, if you have multiple repeating indices, you can just define one set of them and the code will assume that it repeats.
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["#CA0C21","#058A39","#1C397C"]) #DEFINE ORBITALS HERE
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Gamma = [0.0,0.0,0.0]
X = [0.5,0.5,0]
M = [0.5,0,0]
kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
klabels = [r"$\Gamma$","X","M",r"$\Gamma$"] #Write your labels as well. 
Hamplot.BandStructure_orbital_resolved(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100)
Hamplot.ax_BandStructure_orbital.set_title("Maximum orbital Character") #modify the axes using Hamplot.ax_Band
plt.show()
#exit(0)

###############################
#You can also mix the orbital character together to get a better idea of the orbital composition. 
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["#CA0C21","#058A39","#1C397C"],MixOrbitalWeights=True) #DEFINE ORBITALS HERE
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Hamplot.BandStructure_orbital_resolved(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100)
Hamplot.ax_BandStructure_orbital.set_title("Mixed orbital Character") #modify the axes using Hamplot.ax_Band
plt.show()

###############################
#----------------------------------------------------------
##Example 3 - plotting multiple band structures ontop of each other
#----------------------------------------------------------
#Sometimes it's nice to compare different models ontop of each other. All you need to do is create a list of the different models HAM = [HAM1,HAM2,HAM3..] and feed that into Hamiltonian_plot().
#You can make the different models have different colours by specifying a list of colours in Hamplot.BandStructure. 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = -0.3,no_kgrid=True)
Ham2 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham3 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = +0.3,no_kgrid=True)
HAM = [Ham,Ham2,Ham3]
Hamplot = Hamiltonian.tbHamiltonian_Plot(HAM)

Gamma = [0.0,0.0,0.0]
X = [0.5,0.5,0]
M = [0.5,0,0]
kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
klabels = [r"$\Gamma$","X","M",r"$\Gamma$"] #Write your labels as well. 
Hamplot.BandStructure(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100,color=["#680C07","red","blue"])
plt.show()

###############################
#----------------------------------------------------------
##Example 4 - plotting DOS
#----------------------------------------------------------
#the DOS plot has been optimised for speed. it's now very quick.
#Beware the DOS depends on the number of kpoints, and the energy broadening parameter (Gamma). It should converge as k goes to infinity, and Gamma goes to zero, but that would take an infinite amount of calculation time. 
#As a rule of thumb, Gamma (energy broadening) should be approximately equal to the energy step size e.g ((Emax-Emin)/Nsteps)).
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.Load_kgrid(128,128,1) #Choose a k-grid here.

Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
E,DOS = Hamplot.PlotDOS(E_start=-1.0,E_end=+1.0,Nsteps=1000,Gamma=0.01,color="#680C07")
Hamplot.ax_DOS.set_title("DOS example") #modify the axes using Hamplot.ax_DOS
plt.show()

###############################
#----------------------------------------------------------
##Example 5 - plotting orbitally resolved DOS
#----------------------------------------------------------
#Extracting orbital information can be very important. you need to first specify the orbitals present in the model using tbHamiltonian.DefineOrbitals([Orbital labels], [Orbital_Colours])
#Then call PlotDOS_orbital_resolved() which does the same as PlotDOS and takes the same arguements. 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.Load_kgrid(128,128,1)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"],MixOrbitalWeights=True)

Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
E,DOS = Hamplot.PlotDOS_orbital_resolved(E_start=-1.0,E_end=1.0,Nsteps=1000,Gamma=0.01) #You can also save the E and DOS for later plotting/use
Hamplot.ax_DOS.set_title("orbital resolved DOS example") #modify the axes using Hamplot.ax_DOS
plt.show()

###############################
#----------------------------------------------------------
##Example 6 - plotting multiple DOS on top of each other.
#----------------------------------------------------------
#Works in the exact same way as example 2 (plotting band structures on top of each other).  Create an array of the Hamiltonian models and then feed that in. You can change the color by specifying a list of colors to PlotDOS(). 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = -0.3,no_kgrid=True)
Ham.Load_kgrid(32,32,1)
Ham2 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham2.Load_kgrid(32,32,1)
Ham3 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.3,no_kgrid=True)
Ham3.Load_kgrid(32,32,1)

HAM = [Ham,Ham2,Ham3]
Hamplot = Hamiltonian.tbHamiltonian_Plot(HAM)
E,DOS = Hamplot.PlotDOS(E_start=-1.0,E_end=1.0,Nsteps=1000,Gamma=0.01,color=["#680C07","red","blue"]) #You can also save the E and DOS for later plotting/use
Hamplot.ax_DOS.set_title("multiple DOS example") #modify the axes using Hamplot.ax_DOS
plt.show()

###############################
#----------------------------------------------------------
##Example 7 - plotting the Fermi surface (2D)
#----------------------------------------------------------
#Generates a 2D Fermi surface countour, can choose the energy to contour by changing omega.
#Must choose a kz slice to plot.
#Note - this uses matplotlibs plt.contour function. 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.Load_kgrid(128,128,1)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
Hamplot.FermiSurface2D(omega=0.0,kz=0,color="#680C07")
Hamplot.ax_FS.set_title("FS example") #modify the axes using Hamplot.ax_FS
plt.show()

###############################
#----------------------------------------------------------
##Example 8 - plotting orbitally resolved Fermi surface (2D)
#----------------------------------------------------------
#Like with bandstructure and dos, we may also want it orbitally resolved. specify the orbital labels and colours in tbHamiltonian.DefineOrbitals() and then call FermiSurface2D_orbital_resolved() from hamplot.
#HAMPLOT.FermiSurface2D_orbital_resolved() takes all the same arguments as FermiSurface2D.
#Note - this uses my own marching cube algorithm, not the matplotlib plt.contour function. 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.Load_kgrid(128,128,1)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"],MixOrbitalWeights=True) #DEFINE ORBITALS HERE

Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,kz=0)
Hamplot.ax_FS_orbColour.set_title("FS example with orbitals") #modify the axes using Hamplot.ax_FS
plt.show()
Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,kz=0,N_BrillouinZone=2)
Hamplot.ax_FS_orbColour.set_title("ZoomedOut Brillouin Zone") #modify the axes using Hamplot.ax_FS
plt.show()


###############################
#----------------------------------------------------------
##Example 9 - plotting multiple Fermi surfaces (2D)
#----------------------------------------------------------
#Again works in the same way as Example 2 and Example 5. create an array of your Hamiltonian models and the feed that to tbHamiltonian_Plot. 
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = -0.3,no_kgrid=True)
Ham.Load_kgrid(32,32,1)
Ham2 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham2.Load_kgrid(32,32,1)
Ham3 = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.3,no_kgrid=True)
Ham3.Load_kgrid(32,32,1)
HAM = [Ham,Ham2,Ham3]
Hamplot = Hamiltonian.tbHamiltonian_Plot(HAM)
Hamplot.FermiSurface2D(omega=0.0,kz=0,color=["#680C07","red","blue"])
Hamplot.ax_FS.set_title("multiple FS example") #modify the axes using Hamplot.ax_FS
plt.show()


###############################
#----------------------------------------------------------
##Example 10 - plotting the Fermi surface (3D)
#----------------------------------------------------------
#Finally 3D fermi surface plotting in matplotlib. No other random libraries required!
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy =  0.0,no_kgrid=True)
Ham.Load_kgrid(16,16,4)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)
Hamplot.FermiSurface3D(omega=0.0)
Hamplot.ax_FS_3D.set_title("3D Fermi surface!") #modify the axes using Hamplot.ax_FS
plt.show()

###############################
#----------------------------------------------------------
##Example 11 - plotting orbitally resolved Fermi surface (3D)
#----------------------------------------------------------
#Okay i am very happy with this. Same as above, but now 3D Fermi surface.
###############################
Ham = Hamiltonian.tbHamiltonian('Sr214M_hr.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.Load_kgrid(16,16,4)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"],MixOrbitalWeights=True) #DEFINE ORBITALS HERE
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Hamplot.FermiSurface3D_orbital_resolved(omega=0.0)
Hamplot.ax_FS_3D_orb.set_title("3D FS with orbital colour!!!") #modify the axes using Hamplot.ax_FS
plt.show()

exit(0)
###############################
#----------------------------------------------------------
##Example 11 - Calculating the particle number
#----------------------------------------------------------
#We sometimes want to count the number of particles in our system. N = sum_k,sum_bands f(E_band(k)) where f(E_band(k)) is the Fermi function.
#This command does it for you automattically. AS with all the other examples, you can do this either as a total number (needed for e.g fixing the chemical potential) Or as an orbital resolved quantity. 
#Need to check convergence with nk. Should be exact for nk->infinity. 
#The final value in the orbital resolved list is the total particle number. 
###############################
Ham = Hamiltonian.tbHamiltonian('TestData/Sr214M_hr.dat',FermiEnergy = -0.3117)
Ham.Load_kgrid(128,128,1)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"]) #DEFINE ORBITALS HERE

print("Particle number not orbitally resolved:",Ham.CalcParticleNumber(0.0,10)) #energy (eV), temperature (in Kelvin)
print("Particle number, orbital resolved:",Ham.CalcParticleNumber_Orbital_resolved(0.0,10))
print("Density of States At Fermi level:",Ham.CalcDOS(0.0,Gamma=0.005)) #energy (eV) Gamma is an energy braodening (also eV)
print("Density of States At Fermi level:",Ham.CalcDOS_Orbital_resolved(0.0,Gamma=0.005))


###############################
#----------------------------------------------------------
##Example 12 - LATTICE VECTORS
#----------------------------------------------------------
#For anything other than a square lattice. our primative lattice vectors may not be orthogonal. 
#Hexagonal Brillouin zones are a common example.
#Therefore you get everything plotting correctly, you need to include the lattice vectors.
###############################
HAM = Hamiltonian.tbHamiltonian('TestData/graphene_pz_hr.dat',no_kgrid=True)
HAM.DefineOrbitals(["pz"],["red"])
HAM.DefineLatticeVectors([[2.462300,0.000000,0.000000],\
                                  [-1.231150, 2.132414, 0.000000],\
                                  [0.000000, 0.000000, 10.000000]])

#Hexagonal 2D High symmetry points
Gamma = [0.0,0.0,0.0]
M = [0.5,0,0.0] 
K1 = [0.333,0.333,0]
kpath = [Gamma,M,K1,Gamma]
klabels = [r"$/Gamma$","M","K1",r"$/Gamma$"]

HAM.Load_kgrid(16,16,1,symmetry=True) 
HAM.show_Kpoints_and_BrillouinZone() #This shows the Brillouin zone boundary, and the kpoints that will be calculated using Load_kgrid.
HAM.Load_kgrid(16,16,1,symmetry=False)# FermiSurface doesn't yet work with symmetry.  
HAMPLOT = Hamiltonian.tbHamiltonian_Plot(HAM)
HAMPLOT.BandStructure_orbital_resolved(kpath=kpath,klabels=klabels,YMIN=-10,YMAX=15,nk=100)
HAMPLOT.FermiSurface2D_orbital_resolved(1.0,N_BrillouinZone=2,linewidth=6,showBrillouinZone=True,setAntiAliased=False)
HAMPLOT.FermiSurface3D(omega=1.0,N_BrillouinZone_kx=2,N_BrillouinZone_ky=2,N_BrillouinZone_kz=1)
plt.show()


###############################
#----------------------------------------------------------
##Example 13 - Spin orbit coupling
#----------------------------------------------------------
#Given a non-spin orbit coupled Hamiltonian, you can add on-site atomic spin orbit coupling (L.S) using AddSOC(SocValue (eV)).
#To make this work, we need to know the hamiltonian indices of each atom, as well as what orbitals are present at each atom and their relative orientation to the primative lattice vectors. 
#Eventually i want to make this information automatically extractable from the .dat file, but for now you need to specify it manually.
#to do this, create an atom object for each atom in the unit cell.
#the first entry (name) is the name of the atom, it can be anything
#the second entry (position) is the position of the atom in the unit cell. usually found in the .win file.
#the third entry (orbital_character) is the orbital character of the atom. this uses VASP and Quantum Espresso convention 0=s,1=px,2=py,3=pz,4=dz2,5=dxz,6=dyz,7=dx^2-y^2,8=dxy. You can have more than one orbital per atom.
#The fourth entry (hamiltonian_indices) is the index that the orbital character corresponds to in the .dat file. for example in this example we have a dxz,dyz,dxy orbital on the first atom, so the hamiltonian indices are 1,2,3.
#The fifth entry (orientation) is the orientation of the orbitals relative to the primative lattice vectors. for example, in this example the dxz orbitals on both atoms are oriented 45 degrees to the a-lattice vector, so the orientation is [1,-1,0].
#The sixth entry (SOC_LdotS) is the strength of the atomic spin orbit coupling. This can either be a number, or a list [p,d,f] where you can individually specify the strength of the SOC for the p,d or f orbitals. 
#The seventh entry (SOC_Rashba) is the strength of the atomic Rashba spin orbit coupling, defined equivalent to SOC_LdotS. 
###############################

Ham = Hamiltonian.tbHamiltonian('TestData/Sr214M_hr.dat',FermiEnergy = -0.3117,no_kgrid=True)
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"]) #DEFINE ORBITAL colour

#two atom unit cell of Sr2RuO4 containing t2g orbitals, dxz,dyz,dxy. 
Ru1 = Hamiltonian.Atom("Ru1",position=[0.5,0.5,0.5],orbital_character=[5,6,8],hamiltonian_indices=[1,2,3],orientation=[1,0,0],SOC_LdotS = 0.175,SOC_Rashba=0.0)
Ham.DefineAtomicInfo([Ru1]) # name of atom, atomic position, orbitals [0=s,1=px,2=py,3=pz,4=dz2,5=dxz,6=dyz,7=dx^2-y^2,8=dxy],hamiltonian indices

Ham.Load_kgrid(64,64,1)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Gamma = [0.0,0.0,0.0]
X = [0.5,0.5,0]
M = [0.5,0,0]
kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
klabels = [r"$\Gamma$","X","M",r"$\Gamma$"] #Write your labels as well.

Hamplot.BandStructure_orbital_resolved(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100)
Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,kz=0,N_BrillouinZone=2)
plt.show()


###############################
#----------------------------------------------------------
##Example 14 - more complicated Spin orbit coupling
#----------------------------------------------------------
#This should work for any system that has hydrogenic-like spherical harmonic orbitals. 
#Here is an example of a two atom unit cell of Sr2RuO4 containing t2g orbitals, dxz,dyz,dxy at different positions. 
###############################

Ham = Hamiltonian.tbHamiltonian('TestData/Sr2RuO4_0-0_nem0-000.dat',FermiEnergy = 0.0,no_kgrid=True)
Ham.DefineLatticeVectors([[5.4607,    0.0,   0.0],[0.0,    5.4607,   0.0],[0.0,    0.0,   34.1575]]) #can put in lattice vectors or not, up to you for square lattice system. 
Ham.DefineOrbitals(["dxz","dyz","dxy"],["red","green","blue"]) #DEFINE ORBITAL colour

#two atom unit cell of Sr2RuO4 containing t2g orbitals, dxz,dyz,dxy. 
Ru1 = Hamiltonian.Atom("Ru1",position=[0.25,0.75,0.5],orbital_character=[5,6,8],hamiltonian_indices=[1,2,3],orientation=[1,-1,0],SOC_LdotS = 0.175,SOC_Rashba=0.0)
Ru2 = Hamiltonian.Atom("Ru2",position=[0.75,0.25,0.5],orbital_character=[5,6,8],hamiltonian_indices=[4,5,6],orientation=[1,-1,0],SOC_LdotS = 0.175,SOC_Rashba=0.0)
Ham.DefineAtomicInfo([Ru1,Ru2]) # name of atom, atomic position, orbitals [0=s,1=px,2=py,3=pz,4=dz2,5=dxz,6=dyz,7=dx^2-y^2,8=dxy],hamiltonian indices

Ham.Load_kgrid(128,128,1)
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Gamma = [0.0,0.0,0.0]
X = [0.5,0.5,0]
M = [0.5,0,0]
kpath = [Gamma,X,M,Gamma] #input your path as an array. I find it cleanest to write it like this
klabels = [r"$\Gamma$","X","M",r"$\Gamma$"] #Write your labels as well.

Hamplot.BandStructure_orbital_resolved(kpath=kpath,klabels=klabels,YMIN=-1,YMAX=1,nk=100)
Hamplot.FermiSurface2D_orbital_resolved(omega=0.0,kz=0,N_BrillouinZone=2)
plt.show()
#exit(0)

###############################
#----------------------------------------------------------
##Example 14 - Superconducting Hamiltonians
#----------------------------------------------------------
#If you have a tight binding model that's superconducting. We can set Tc via Ham.Tc = 10 and modify the temperature via Ham.ModifySuperconductingGap(T)
#You have to explicitely have a superconducting hamiltonian in the .dat file and specify superconducting=True in the tbHamiltonian object.
#You can then set a Tc value (in Kelvin) using e.g Ham.Tc = 10. By default Ham.Tc = 1.
#For studying the temperature dependence of the superconducting gap, you can use Ham.ModifySuperconductingGap(T) where T is the temperature in Kelvin. This modifies the gap size specified in the .dat file using the BCS formula Delta_0*(tanh(sqrt(Tc/T - 1)) and will set the gap to zero at T=Tc. 
###############################

Ham_SC = Hamiltonian.tbHamiltonian('TestData/2D_squareLattice-dwave.dat',FermiEnergy =0.05,no_kgrid=True,superconducting=True) # Make sure to set superconducting =True.
Ham_SC.Tc = 15 # gap in 2D_squareLattice is 1.76*kb*Tc = 0.002275 eV <-This is large on purpose to make the effect visible.
nk = 512
for T in [1,5,10,13,14,15,16,20]:
    Ham_SC.ModifySuperconductingGap(T) #Put Temperature in here. 
    Ham_SC.Load_kgrid(nk,nk,1,symmetry=True,Kpoints_only=False) #Calculate the eigenvalues/
    Hamplot_SC = Hamiltonian.tbHamiltonian_Plot(Ham_SC)
    Hamplot_SC.PlotDOS(E_start=-0.02,E_end=+0.02,Nsteps=1000,Gamma=0.001,color="#680C07")
    Hamplot_SC.ax_DOS.set_title(f"T = {T}K, Tc= 15K") #modify the axes using Hamplot.ax_DOS

plt.show()

###############################
#----------------------------------------------------------
##Example 15 - Specific Heat
#----------------------------------------------------------
#Calculating the specific heat as a function of temperature. 
#returns a list of Tempeartures and specific heat values. These are x and y axes. 
#if DivideByT = True, it will plot/save C/T instead of C.

#In the case of a superconducting Hamiltonian, it will calculate the specific heat for both the superconducting and normal state structures, you should see a jump at T=Tc. 

#For the specific heat we aren't able to calculate the entire eigenvalue grid in advance, (due to sub-optimal coding) For this reason, you can specify Kpoints_only=True in Ham.Load_kgrid() to only calculate the kpoints and save double counting. 

# NOTE - For very small temperatures, a large number of k-points is required. else it might just go to zero due to numerical errors.
###############################

Ham = Hamiltonian.tbHamiltonian('TestData/Sr214M_hr.dat',FermiEnergy = -0.3117,no_kgrid=True)
Ham.Load_kgrid(128,128,1,symmetry=True,Kpoints_only=True) #Choose a k-grid here.
Hamplot = Hamiltonian.tbHamiltonian_Plot(Ham)

Temp,Clist = Hamplot.PlotSpecificHeat(T_start=0.1,T_end=20,T_step=0.1,DivideByT=True) #DivideByT = True will plot C/T instead of C.

Ham_SC = Hamiltonian.tbHamiltonian('TestData/2D_squareLattice-dwave.dat',FermiEnergy =0.05,no_kgrid=True,superconducting=True) # Make sure to set superconducting =True.
Ham_SC.Tc = 15 # gap in 2D_squareLattice is 1.76*kb*Tc = 0.002275 eV <-This is large on purpose to make the effect visible.
Ham_SC.Load_kgrid(128,128,1,symmetry=True,Kpoints_only=True) #Choose a k-grid here.
Hamplot_SC = Hamiltonian.tbHamiltonian_Plot(Ham_SC)
Temp,Clist = Hamplot_SC.PlotSpecificHeat(T_start=0.1,T_end=20,T_step=0.1,DivideByT=True) #DivideByT = True will plot C/T instead of C.

plt.show()





