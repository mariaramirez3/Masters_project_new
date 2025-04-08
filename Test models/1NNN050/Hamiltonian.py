import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import linalg as LA
import math
from shapely.geometry import Point, Polygon

import numba as nb 

import time  
        
def timeit(func): #decorator that allows us to determine speeds of functions.
    def wrapper(*args,**kwargs):
        startTime =time.time()
        Val = func(*args,**kwargs)
        timeTaken = time.time()-startTime
        print(func.__name__,'took: ',timeTaken,'s')
        return Val
    return wrapper

class tbHamiltonian():

    r'''
    This class is used to load and store the tight binding Hamiltonian. It can also be used to load the eigenvalues and eigenvectors over a k-grid.
    '''

    def __init__(self, filename,FermiEnergy=0.0,no_kgrid=False,IsSpinPolarised=False,hasSOC=False,hasRashbaSOC=False,superconducting=False):
        self.path = filename
        self.filename = filename.split("/")[-1]


        self.nkx = 0
        self.nky = 0
        self.nkz = 0
        self.nbands = 0
        self.no_kgrid = no_kgrid
        self.DOS = []
        self.DOSatEF = None
        self.Orbital_Label = []
        self.Orbital_Colour = []
        self.MixOrbitalWeights = None
        self.Orbital_Repetition = 0
        self.Eigenvalues,self.EigenVectors = None,None
        self.lattice_vectors = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]],dtype=np.float64)
        self.reciprocal_vectors = self.get_reciprocal_vectors(self.lattice_vectors)
        self.BZcorners,self.BZedges,self.BZfaces = self.MakeBZBoundaries() #MakeBZBoundaries(self.lattice_vectors)
        #self.BZplanes,self.BZcorners,self.BZedges = MakeBZBoundaries(self.lattice_vectors)#MakeBZBoundaries(self.lattice_vectors)
        self.lattpg_op = get_lattice_pointGroup(self.lattice_vectors, eps=1E-10)
        self.BZcorners_irreducible,self.BZedges_irreducible = MakeIrreducibleBZBoundaries(self.lattpg_op,self.BZcorners,self.BZedges)
        self.Kpoints_BZ = None
        self.Kpoints_IBZ = None
        self.symmetry = False
        #print(self.BZedges)
        self.nkx=0
        self.nky=0
        self.nkz=0
        self.EigenvaluesOnly = False

        self.Tc = 1.0

        self.hopping_parameters = []
        self.GapRemoval = []
        self.R_degeneracy = []
        self.SOChopping_parameters = [] #SOC matrices also written in same format as hopping parameters. L dot S
        self.RashbaSOChopping_parameters = [] # Rashba SOC matrices also written in same format as hopping parameters.  Lx sigma_y - Ly sigma_x
        self.FermiEnergy_hoppings =[]

        self.SpinPolarised = IsSpinPolarised
        self.SOC = hasSOC
        self.SOCstrength = 0.0
        self.RashbaSOC = hasRashbaSOC
        self.RashbaSOCstrength = 0.0
        self.ExchangeSplitting = 0.0
        self.Superconducting = superconducting
        self.CurrentTemperature = 1

        self.read_hoppings() #This needs to be exactly here. Above Fermi energy, but below everything else.

        #self.nbands and self.hopping_parameters are defined here.
        self.FermiEnergy = 0
        self.SetFermiEnergy(FermiEnergy)

        self.atomic_info = []

    def read_hoppings(self):
        """
        This function reads a Wannier90 formatted tight binding model file. Often saved as XXX_hr.dat.
        If this doesn't work, nothing will.
        importantly it reads 
         - self.nbands : number of bands/orbitals in the model (size of the hamiltonian).
         - self.hopping_parameters : a list of [Rx,Ry,Rz,orbital_index_1,orbital_index_2,Re(t),Im(t),Rdegeneracy] where Re(t) and Im(t) are the real and imaginary parts of the hopping parameter that connecting two orbitals seperated by a unit cell vector R = [Rx,Ry,Rz]. 
        Unfortunately, R_degeneracy is needed in Wannier90. You have to divide each hopping parameter by the R_degeneracy value that corresponds to the specific R-vector.
        see e.g http://www.wanniertools.com/input.html#wannier90-dat
        """
        self.hopping_parameters = []
        self.R_degeneracy = []
        f = open(self.path, "r")
        count = 0
        PASS = False
        Rlast=0
        R_counter=0
        for line in f:
            l = line.split()
            #print(count,l)
            if count == 1:
                self.nbands = int(l[0]) #number of bands/Hamiltonian size
            elif count == 2:
                self.Rnum = int(l[0]) # number of unique R-vectors
            elif count ==3:
                lenl = len(l)
                for i in l:
                    self.R_degeneracy.append(float(i)) # This is the list of numbers usually in rows of 15 numbers. 

            elif count > 3:
                if (len(self.R_degeneracy) < self.Rnum):
                    for i in l:
                        self.R_degeneracy.append(float(i))
                else:
                    if len(l) > 0:
                        #Rx Ry Rz s p tr ti
                        #This is to determine what the R-degeneracy of the hopping parameter is.
                        if Rlast ==0:
                            Rlast = [float(l[0]),float(l[1]),float(l[2])]
                        R = [float(l[0]),float(l[1]),float(l[2])]

                        if R != Rlast:
                            R_counter +=1
                            Rlast = R

                        if ((float(l[5]) != 0) or (float(l[6]) !=0)): #Dont add in zeros, that just slows things down.
                            self.hopping_parameters.append([float(l[0]),float(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6]),self.R_degeneracy[R_counter]])
                        #print(HAM[-1])
            count += 1
        self.hopping_parameters = self.hopping_parameters
        self.R_degeneracy = self.R_degeneracy

    
    def DefineOrbitals(self,OrbitalLabel,OrbitalColour,MixOrbitalWeights=None):
        """
        If you have a model with multiple orbitals, you can give each orbital index a label and colour using this function.
        This is needed if you want to plot orbital resolved functions. You will recieve an error if you have not called this function. 
        Eventually I will make it so that the initial Hamiltonian text file will contain this information. 

        If you have multiple repeated orbital indices. (e.g 2 atoms at different positions with the same orbitals) just write it out once and then the code will repeat it.  
        example: OrbitalLabel = [dxz,dyz,dxy] OrbitalColour = [red,blue,green] would work for a one atom three band model, or a two atom six band model, where index 1 and 4 will be "dxz" and coloured red. 

        MixOrbitalWeights tells the rest of the code whether you want to display the Maximum orbital character of e.g the bandstructure. Or if you want to mix the colours together to get a smoother evolution of the colours. 
        """
        #check orbitalLabel and ORbital Color are lists of the same length.
        if len(OrbitalLabel) != len(OrbitalColour):
            print("OrbitalLabel and OrbitalColor must be lists of the same length")
            return 0
        #check the list supplied is a multiple of the number of bands.
        if self.nbands%len(OrbitalLabel) != 0:
            print("OrbitalLabel and OrbitalColor must be the same size as either the number of orbitals, or some fraction of the number of orbitals")
            print("Number of bands / length of OrbitalLabel = ",self.nbands/len(OrbitalLabel))
            return 0
        if len(self.Orbital_Colour) != 0:
            self.Orbital_Colour.clear
            self.Orbital_Label.clear
            self.Orbital_Repetition = 0
        else:
            for orb_colour in OrbitalColour: # Convert to tuple
                self.Orbital_Colour.append(colors.to_rgb(orb_colour))

            self.Orbital_Repetition = self.nbands/len(OrbitalLabel)
            self.Orbital_Label = OrbitalLabel
            if MixOrbitalWeights != None:
                self.MixOrbitalWeights = True

    def DefineLatticeVectors(self,lattice_vectors):
        """
        This function is used to define the lattice vectors of the system.
        If not called the system assumes cubic lattice vectors with the lattice constant set to 1. 
        If you define this, you can use symmetry to identify the irreducible brillouin zone (thus speeding up calculations) and will get the correct Brillouin zone boundaries.
        You can check this is correct by using self.show_Kpoints_and_BrillouinZone() and checking the plot. 

        Lattice vectors should be a 3x3 matrix of the form [[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]] where a,b,c are the three lattice vectors.
        """
        if type(lattice_vectors) == list:
            lattice_vectors = np.array(lattice_vectors,dtype=np.float64)
        
        if type(lattice_vectors) == np.ndarray:
            if lattice_vectors.shape == (2,2): #can put in 2D array, but we'll just extend to 3D
                new_lattice_vectors = np.zeros((3,3),dtype=np.float64) 
                for i in range(2):
                    for j in range(2):
                        new_lattice_vectors[i][j] = lattice_vectors[i][j]

            if lattice_vectors.shape == (3,3):
                self.lattice_vectors = np.array(lattice_vectors,dtype=np.float64)
                self.reciprocal_vectors = self.get_reciprocal_vectors(self.lattice_vectors)
                #self.BZcorners,self.BZedges = MakeBZBoundaries(self.lattice_vectors)
                self.BZcorners,self.BZedges,self.BZfaces = self.MakeBZBoundaries() #MakeBZBoundaries(self.lattice_vectors)

                self.lattpg_op = get_lattice_pointGroup(self.lattice_vectors, eps=1E-10)
                self.BZcorners_irreducible,self.BZedges_irreducible = MakeIrreducibleBZBoundaries(self.lattpg_op,self.BZcorners,self.BZedges)
    
    
    def MakeBZBoundaries(self):
        """
        This calls a bunch of nasty functions that use the lattice vectors defined above and calculate the Brillouin zone boundaries as a list of corners, edges and faces.
        corners are simply a list of points [kx,ky,kz] for each corner of the Brillouin zone. 
        edges are lines [kx_start,ky_start,kz_start,kx_end,ky_end,kz_end] for each boundary line in the Brillouin zone. The first three entry is the starting point, the final three entries is the end point. 
        faces are the planes that make up the Brillouin zone. I forget how this works. 
        """
        G,G1 = calculateBZPlanes(self.lattice_vectors) 
        corners = calculateBZCorners(G,G1)
        edges = calculateBZEdges(G1,corners)
        faces = calculateBZFaces(G1,corners)

        return corners,edges,faces
 
               
    def DefineAtomicInfo(self,atoms):
        """
        If you have atoms with different positions and orbital characters, you can define them here. 
        Ideally I want this to be included in the hamiltonian input file. But for now that's not the case. 
        
        #This needs to be a list of list [[atom, [x,y,z],[orbital_index_1...orbital_index_n],[hamiltonian_index_1...hamiltonian_index_n]]
        #atom is a string that defines the name of the atom, can be anything you like. 
        #x,y,z are the positions of the atom in the unit cell, in units of the lattice vectors.
        #orbital_index_1...orbital_index_n are the orbital indices that correspond to the atom. # 0 s 1 px 2 py 3 pz 4 dz2 5 dxz 6 dyz 7 dx2-y2 8 dxy
        #hamiltonian_index_1...hamiltonian_index_n are the indices of the hopping parameters that correspond to the orbital. 
        # and example would be
        #[[Ru, [0.5,0.5,0.5],[5,6,8],[1,2,3]]] for a 3 band model of a single Ru atom at 0.5,0.5,0.5 with dxz,dyz,dxy orbitals.
         
        """
        if isinstance(atoms,Atom):
            try: atoms = [atoms]
            except:
                print("atoms must be a list:",atoms)
                return 0
        
        if isinstance(atoms,list):
            if not isinstance(atoms[0],Atom):
                print("atoms needs to be a list of Hamiltonian.Atom objects")
                exit(0)
        
        self.atomic_info = atoms
        
        #If SOC is finite on any orbital, add SOC to the model.
        hasSOC_LdotS = False
        hasSOC_Rashba = False
        for atom in atoms:
            if atom.SOC_LdotS != [0.0,0.0,0.0]:
                hasSOC_LdotS = True
            if atom.SOC_Rashba != [0.0,0.0,0.0]:
                hasSOC_Rashba = True
        if hasSOC_LdotS == True:
            print("Adding SOC to the model")
            self.AddSOC()
        if hasSOC_Rashba == True:
            self.AddRashbaSOC()

    def MakeSpinPolarised(self):
        """
        Takes the model, checks if it's spin polarised, if not, doubles the number of bands and adds the spin polarised hopping parameters to the spin down channel.

        i.e takes H(k) and converts it into a 2x2 matrix of [[H(k)^up 0], [0, H(k)^down]].
        """
        if self.SpinPolarised == False:
            self.SpinPolarised = True
            #This makes the spin down block. 
            hopping_parameters = []
            for i in self.hopping_parameters:
                hopping_parameters.append([i[0],i[1],i[2],i[3]+self.nbands,i[4]+self.nbands,i[5],i[6],i[7]]) #Rx,Ry,Rz,s+nbands,p+nbands,tr,ti, Rdegeneracy (Rdegenearcy needed for Wannier90, but soon to become redundant)
            self.hopping_parameters = self.hopping_parameters+ hopping_parameters
            for i in range(self.nbands):
                self.FermiEnergy_hoppings.append([0.,0.,0.,i+self.nbands+1,i+self.nbands+1,-self.FermiEnergy,0.0,1.0])
            self.nbands = 2*self.nbands
            
            
            
    
    def AddSOC(self):
        """
        #Takes a Hamiltonian, and adds atomic spin orbit coupling via. L.S 
        #Requires self.atom_info to be set. 
        #Hopefully this will automatically be done in the future hamiltonian files. 
        #Also can take into account that orbitals may not be positioned along the x,y,z axis of the unit cell. 
        """

        if self.SOC == False:
            self.SOC = True
            if self.SpinPolarised == False:
                self.MakeSpinPolarised()
        #print("Done")
        
        SOChopping_parameters = []
            
        #self.SOCstrength = SOC_strength
        #SOC can be for any orbital pointing along any axis. need to take care of that by doing a rotation of the Lx,Ly and Lz matrices which are defined with respect to the a,b and c lattice directions. 
        for atom in self.atomic_info: #atom info should be atom_info[atom_index][0] = atom_name [1] = atom positions, [2] = orbital indices, [3] = hamiltonian indices
            #theta = np.pi / 4  # Example rotation angle
            v = np.array([1,0, 0])  #  x- axis to get in plane theta angle 
            w = np.array([atom.orientation[0],atom.orientation[1],0.0])  # project onto xy plane. 
            cos_theta = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            theta_deg = np.degrees(theta_rad)
            #print(theta_deg)
            v = np.array([0,0, 1])  #  z- axis
            w = np.array([atom.orientation[0],0,atom.orientation[2]])  # project onto xz plane.
            cos_phi = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
            phi_rad = np.arccos(np.clip(cos_phi, -1.0, 1.0))
            phi_deg = np.degrees(phi_rad)
            #print(phi_deg)
            #exit(0)
            #First rotate the L angular momentum matrices with respect to the x-axis
            R_theta = np.array([[np.cos(theta_rad), -np.sin(theta_rad),0],
                        [np.sin(theta_rad), np.cos(theta_rad),0],
                        [0,0,1]])
            #Then rotate the L angular momentum matrices out of the xy plane. the -np.pi/2.0 is to rotate with respect to the xy plane. 
            R_phi = np.array([[np.cos(phi_rad-np.pi/2.0), 0, -np.sin(phi_rad-np.pi/2.0)], #
                        [0, 1, 0],
                        [np.sin(phi_rad-np.pi/2.0), 0, np.cos(phi_rad-np.pi/2.0)]])

            
            xxp,xyp,xzp = np.dot(R_phi,np.dot(R_theta,[1,0,0]))
            yxp,yyp,yzp = np.dot(R_phi,np.dot(R_theta,[0,1,0]))
            zxp,zyp,zzp = np.dot(R_phi,np.dot(R_theta,[0,0,1]))
            #print(xxp,xyp,xzp)
            #print(yxp,yyp,yzp)
            #print(zxp,zyp,zzp)
            #exit(0)
            
            Lx_d_rotated = xxp*Lx_d + xyp*Ly_d + xzp*Lz_d
            Ly_d_rotated = yxp*Lx_d + yyp*Ly_d + yzp*Lz_d
            Lz_d_rotated = zxp*Lx_d + zyp*Ly_d + zzp*Lz_d

            Lx_p_rotated = xxp*Lx_p + xyp*Ly_p + xzp*Lz_p
            Ly_p_rotated = yxp*Lx_p + yyp*Ly_p + yzp*Lz_p
            Lz_p_rotated = zxp*Lx_p + zyp*Ly_p + zzp*Lz_p
        
            
            for orb1_i,orb1 in enumerate(atom.orbital_character): #This should be non SOC indices. 
                for orb2_i,orb2 in enumerate(atom.orbital_character):
                    if (orb1 >3) and (orb2 >3): #D ORBITALS
                        #LZ
                        if np.abs(Lz_d_rotated[orb1-4][orb2-4]) > 0.0:
                            L = (atom.SOC_LdotS[1]/2.0)*Lz_d_rotated[orb1-4][orb2-4] #atom.SOC_LdotS[1] is the SOC strength for the d orbitals of this particular atom
                            Ham_index_1 = atom.hamiltonian_indices[orb1_i]
                            Ham_index_2 = atom.hamiltonian_indices[orb2_i]
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1,Ham_index_2,L.real,L.imag,1.0]) #H_up_up
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1+int(self.nbands/2.0),Ham_index_2+int(self.nbands/2.0),-L.real,-L.imag,1.0])  #H_down_down
                        #LX +iLY 
                        if (np.abs(Lx_d_rotated[orb1-4][orb2-4]) > 0.0) or np.abs(Ly_d_rotated[orb1-4][orb2-4]) > 0.0:
                            L_updown = (atom.SOC_LdotS[1]/2.0)*(Lx_d_rotated[orb1-4][orb2-4] + 1j*Ly_d_rotated[orb1-4][orb2-4])
                            L_downup = (atom.SOC_LdotS[1]/2.0)*(Lx_d_rotated[orb1-4][orb2-4] - 1j*Ly_d_rotated[orb1-4][orb2-4]) #Hermitian conjugate
                            Ham_index_1 = atom.hamiltonian_indices[orb1_i]
                            Ham_index_2 = atom.hamiltonian_indices[orb2_i]
                            #Check this is the right way round!
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1,Ham_index_2+int(self.nbands/2.0),L_updown.real,L_updown.imag,1.0]) 
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1+int(self.nbands/2.0),Ham_index_2,L_downup.real,L_downup.imag,1.0]) 
                    elif (0 <orb1 < 4 ) and (0 < orb2 < 4): #P ORBITALS
                        #LZ
                        if np.abs(Lz_p_rotated[orb1-1][orb2-1]) > 0.0:
                            L = (atom.SOC_LdotS[0]/2.0)*Lz_p_rotated[orb1-1][orb2-1] #atom.SOC_LdotS[0] is the SOC strength for the p orbitals of this particular atom
                            Ham_index_1 = atom.hamiltonian_indices[orb1_i]
                            Ham_index_2 = atom.hamiltonian_indices[orb2_i]
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1,Ham_index_2,L.real,L.imag,1.0]) #H_up_up
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1+int(self.nbands/2.0),Ham_index_2+int(self.nbands/2.0),-L.real,-L.imag,1.0])  #H_down_down
                        #LX +iLY 
                        if (np.abs(Lx_p_rotated[orb1-1][orb2-1]) > 0.0) or np.abs(Ly_p_rotated[orb1-1][orb2-1]) > 0.0:
                            L_updown = (atom.SOC_LdotS[0]/2.0)*(Lx_p_rotated[orb1-1][orb2-1] + 1j*Ly_p_rotated[orb1-1][orb2-1])
                            L_downup = (atom.SOC_LdotS[0]/2.0)*(Lx_p_rotated[orb1-1][orb2-1] - 1j*Ly_p_rotated[orb1-1][orb2-1]) #Hermitian conjugate
                            Ham_index_1 = atom.hamiltonian_indices[orb1_i]
                            Ham_index_2 = atom.hamiltonian_indices[orb2_i]
                            #Check this is the right way round!
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1,Ham_index_2+int(self.nbands/2.0),L_updown.real,L_updown.imag,1.0]) 
                            SOChopping_parameters.append([0.,0.,0.,Ham_index_1+int(self.nbands/2.0),Ham_index_2,L_downup.real,L_downup.imag,1.0]) 
                    
        self.SOChopping_parameters = SOChopping_parameters
        #print(self.SOChopping_parameters)

    def AddRashbaSOC(self):
        """
        #Takes a Hamiltonian, and adds atomic spin orbit coupling via. L.S 
        #Requires self.atom_info to be set. 
        #Hopefully this will automatically be done in the future hamiltonian files. 
        #Also can take into account that orbitals may not be positioned along the x,y,z axis of the unit cell. 
        """

        if self.RashbaSOC == False:
            self.RashbaSOC = True
            if self.SpinPolarised == False:
                self.MakeSpinPolarised()

        RashbaSOChopping_parameters = []
            
        #self.RashbaSOCstrength = SOC_strength

        for atom in self.atomic_info: #atom info should be atom_info[atom_index][0] = atom_name [1] = atom positions, [2] = orbital indices, [3] = hamiltonian indices
            #theta = np.pi / 4  # Example rotation angle
            v = np.array([1,0, 0])  #  x- axis to get in plane theta angle 
            w = np.array([atom.orientation[0],atom.orientation[1],0.0])  # project onto xy plane. 
            cos_theta = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            theta_deg = np.degrees(theta_rad)
            #print(theta_deg)
            v = np.array([0,0, 1])  #  z- axis
            w = np.array([atom.orientation[0],0,atom.orientation[2]])  # project onto xz plane.
            cos_phi = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
            phi_rad = np.arccos(np.clip(cos_phi, -1.0, 1.0))
            phi_deg = np.degrees(phi_rad)
            #exit(0)
            #First rotate the L angular momentum matrices with respect to the x-axis
            R_theta = np.array([[np.cos(theta_rad), -np.sin(theta_rad),0],
                        [np.sin(theta_rad), np.cos(theta_rad),0],
                        [0,0,1]])
            #Then rotate the L angular momentum matrices out of the xy plane. the -np.pi/2.0 is to rotate with respect to the xy plane. 
            R_phi = np.array([[np.cos(phi_rad-np.pi/2.0), 0, -np.sin(phi_rad-np.pi/2.0)], #
                        [0, 1, 0],
                        [np.sin(phi_rad-np.pi/2.0), 0, np.cos(phi_rad-np.pi/2.0)]])


            xxp,xyp,xzp = np.dot(R_phi,np.dot(R_theta,[1,0,0]))
            yxp,yyp,yzp = np.dot(R_phi,np.dot(R_theta,[0,1,0]))
            zxp,zyp,zzp = np.dot(R_phi,np.dot(R_theta,[0,0,1]))

            Sigma_x_rotated = xxp*Sigma_x + xyp*Sigma_y + xzp*Sigma_z
            Sigma_y_rotated = yxp*Sigma_x + yyp*Sigma_y + yzp*Sigma_z
            Sigma_z_rotated = zxp*Sigma_x + zyp*Sigma_y + zzp*Sigma_z

            for orb1_i,orb1 in enumerate(atom.orbital_character): #This should be non SOC indices.
                Ham_index = atom.hamiltonian_indices[orb1_i]
                Rashba_sigmax_01 = -1j*(atom.SOC_Rashba[1]/2.0)*Sigma_x_rotated[0,1]
                Rashba_sigmax_10 = -1j*(atom.SOC_Rashba[1]/2.0)*Sigma_x_rotated[1,0]
                #+dy sigma_x  
                RashbaSOChopping_parameters.append([0,1,0.,Ham_index,Ham_index+int(self.nbands/2.0),Rashba_sigmax_01.real,Rashba_sigmax_01.imag,1.0])
                RashbaSOChopping_parameters.append([0,1,0.,Ham_index+int(self.nbands/2.0),Ham_index,Rashba_sigmax_10.real,Rashba_sigmax_10.imag,1.0])
                #-dy sigma_x  
                RashbaSOChopping_parameters.append([0,-1,0.,Ham_index,Ham_index+int(self.nbands/2.0),-Rashba_sigmax_01.real,-Rashba_sigmax_01.imag,1.0])
                RashbaSOChopping_parameters.append([0,-1,0.,Ham_index+int(self.nbands/2.0),Ham_index,-Rashba_sigmax_10.real,-Rashba_sigmax_10.imag,1.0]) 

                #dx sigma_y
                Rashba_sigmay_01 = -1j*(atom.SOC_Rashba[1]/2.0)*Sigma_y_rotated[0,1]
                Rashba_sigmay_10 = -1j*(atom.SOC_Rashba[1]/2.0)*Sigma_y_rotated[1,0]
                RashbaSOChopping_parameters.append([ 1,0,0.,Ham_index,Ham_index+int(self.nbands/2.0),Rashba_sigmay_01.real,Rashba_sigmay_01.imag,1.0])
                RashbaSOChopping_parameters.append([ 1,0,0.,Ham_index+int(self.nbands/2.0),Ham_index,Rashba_sigmay_10.real,Rashba_sigmay_10.imag,1.0]) 

                RashbaSOChopping_parameters.append([-1,0.,0.,Ham_index,Ham_index+int(self.nbands/2.0),-Rashba_sigmay_01.real,-Rashba_sigmay_01.imag,1.0]) 
                RashbaSOChopping_parameters.append([-1,0.,0.,Ham_index+int(self.nbands/2.0),Ham_index,-Rashba_sigmay_10.real,-Rashba_sigmay_10.imag,1.0])  

            self.RashbaSOChopping_parameters = RashbaSOChopping_parameters
        


    def SetFermiEnergy(self,FermiEnergy):
        """
        allows you to change the checical potential.
        """
        self.FermiEnergy = FermiEnergy
        if self.Superconducting == True:
            half_nbands = int(self.nbands/2)
            for i in range(half_nbands):
                self.FermiEnergy_hoppings.append([0.,0.,0.,i+1,i+1,FermiEnergy,0.0,1.0])
                self.FermiEnergy_hoppings.append([0.,0.,0.,i+1+half_nbands,i+1+half_nbands,-FermiEnergy,0.0,1.0])
        else:
            for i in range(self.nbands):
                self.FermiEnergy_hoppings.append([0.,0.,0.,i+1,i+1,-FermiEnergy,0.0,1.0])

 
    def get_reciprocal_vectors(self,lattice_vectors):
        return 2*np.pi*LA.inv(lattice_vectors).T
    
    def show_Kpoints_and_BrillouinZone(self):
        """
        Function that plots the Brillouin zone, the irreducible Brillouin zone (if symmetry is defined), and the k-points spaced over the grid.
        This allows you to perform a check that you're lattice parameters and k-points are correct. 
        For plotting purposes it's best to have only load a small number of k-points e.g 16x16x1. 
        """
        if (self.nkx == 0) or (self.nky == 0) or (self.nkz==0):
            print("No k-grid loaded. Please load a k-grid using tbHamiltonian.Load_kgrid(nkx,nky,nkz) or specify no_kgrid=True in tbHamiltonian()")
            print("going to use automatic kgrid of 16x16x1")
            self.Load_kgrid(16,16,1)

        fig = plt.figure()
        if self.nkz == 1:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        
        for k in self.Kpoints_BZ:
            if self.nkz == 1:
                ax.scatter(k[0],k[1],marker='o',c="black")
            else:
                ax.scatter(k[0],k[1],k[2],marker='o',c="black")
            #plt.scatter(kx,ky,c="black")
        
        if self.Kpoints_IBZ != None:
            for k in self.Kpoints_IBZ:
                if self.nkz == 1:
                    ax.scatter(k[0],k[1],marker='x',c="red")
                else:
                    ax.scatter(k[0],k[1],k[2],marker='x',c="red")
                #plt.scatter(kx,ky,c="black")
                
        for corner in self.BZcorners:
            #print("corner",corner)
            if self.nkz == 1:
                ax.scatter(corner[0],corner[1],marker='o',c="red")
                #print(corner[0],corner[1])
                #ax.text(np.round(corner[0],3),np.round(corner[1],3),f"{corner[0]},{corner[1]}")
            else:
                ax.scatter(corner[0],corner[1],corner[2],marker='o',c="red")
            
        for edge in self.BZedges:
            #print("edge",edge)
            x= np.linspace(edge[0],edge[3],2)
            y = np.linspace(edge[1],edge[4],2)
            z = np.linspace(edge[2],edge[5],2)
            if self.nkz == 1:
                ax.plot(x,y,c="black")
            else:
                ax.plot(x,y,z,c="black")

        if self.BZcorners_irreducible != None:
            for corner in self.BZcorners_irreducible:
                if self.nkz == 1:
                    ax.scatter(corner[0],corner[1],marker='o',c="blue")
                    #print(corner[0],corner[1])
                    #ax.text(np.round(corner[0],3),np.round(corner[1],3),f"{corner[0]},{corner[1]}")
                else:
                    ax.scatter(corner[0],corner[1],corner[2],marker='o',c="blue")
            
            for edge in self.BZedges_irreducible:
                x= np.linspace(edge[0],edge[3],2)
                y = np.linspace(edge[1],edge[4],2)
                z = np.linspace(edge[2],edge[5],2)
                if self.nkz == 1:
                    ax.plot(x,y,c="red")
                else:
                    ax.plot(x,y,z,c="red")

        max_blength = 0
        for b in self.reciprocal_vectors:
            if self.nkz == 1:
                ax.quiver(0,0,b[0],b[1],angles='xy',scale_units='xy',scale=1)
                blength = np.sqrt(b[0]**2+b[1]**2)
                if max_blength < blength:
                    max_blength = blength
            else:
                ax.quiver(0,0,0,b[0],b[1],b[2],arrow_length_ratio=0.1)
                blength = np.sqrt(b[0]**2+b[1]**2+b[2]**2)
                if max_blength < blength:
                    max_blength = blength
        

        ax.set_xlim([-max_blength,max_blength])
        ax.set_ylim([-max_blength,max_blength])
        #ax.set_xlim([-1.5*max_blength,1.5*max_blength])
        #ax.set_ylim([1.5*max_blength,1.5*max_blength])
        if self.nkz != 1:
            ax.set_zlim([-max_blength,max_blength])
            ax.set_aspect('auto')
        else:
            ax.set_aspect('equal')
        plt.show()

    #Calculates the orbital weights for the specified orbital_Labels. 
    def GetOrbitalWeights(self,eigenvalue_index,eigenvector):
        """
        If orbital labels are defined, this function will return the orbital weights for the specified eigenvalue index and eigenvector.
        This is used to get e.g the colour of the bandstructure at a specific k-point.
        """
        Weights = calcOrbitalWeights(eigenvalue_index,eigenvector)

        if self.Orbital_Repetition !=0:
            Orbital_weights = [0 for i in range(len(self.Orbital_Colour))]
            for i in range(len(eigenvector)): #every orbital index
                Orbital_weights[i%len(self.Orbital_Colour)] += Weights[i]
            return Orbital_weights
        else:
            return Weights
    
    def GetHoppings(self):
        """
        Takes all the different terms, e.g spin orbit coupling, superconductivity, chemical potential, which go into the Hamiltonian and returns the output as a single list to be used in H(k) = sum_R e^(i k.R) t(R)
        """

        return np.array(self.hopping_parameters+self.SOChopping_parameters+self.GapRemoval + self.FermiEnergy_hoppings + self.RashbaSOChopping_parameters,dtype=np.float32)

    #@timeit
    def Load_kgrid(self,nkx,nky,nkz,symmetry=False,Kpoints_only=False):
        """
        Loads the eigenvalues and eigenvectors for a k-grid of size nkx,nky,nkz. These points are spaced using the Monkhurst-pack formula.
        The K-points evenly spaced over the first Brillouin zone using the Monkhurst-Pack formula is first obtained and stored. 
        If symmetry is true it will then calculate the irreducible set of k-points. 

        If Kpoints_only is true, it will only calculate the k-points and not the eigenvalues and eigenvectors.

        """
        self.symmetry=symmetry

        if Kpoints_only == True:
            if (self.nkx != nkx) or (self.nky != nky) or (self.nkz != nkz):
                self.Kpoints_BZ = Load_Kpoints_BZ(nkx,nky,nkz,self.reciprocal_vectors)
            if symmetry==True:
                if self.BZcorners_irreducible != None:
                    if self.nkx != nkx or self.nky != nky or self.nkz != nkz:
                        self.Kpoints_IBZ = Calculate_kpoints_IBZ(self.Kpoints_BZ,self.reciprocal_vectors,self.BZcorners_irreducible)
                    #self.Weights = Calculate_kpoints_IBZ_Weights(self.Kpoints_IBZ,self)
        else:
            if (self.nkx != nkx) or (self.nky != nky) or (self.nkz != nkz):
                self.Kpoints_BZ = Load_Kpoints_BZ(nkx,nky,nkz,self.reciprocal_vectors)
            if symmetry==True:
                if self.BZcorners_irreducible != None:
                    if self.nkx != nkx or self.nky != nky or self.nkz != nkz:
                        self.Kpoints_IBZ = Calculate_kpoints_IBZ(self.Kpoints_BZ,self.reciprocal_vectors,self.BZcorners_irreducible)
                    #self.Weights = Calculate_kpoints_IBZ_Weights(self.Kpoints_IBZ,self)
                if self.EigenvaluesOnly == True:
                    self.Eigenvalues = Load_Hamiltonian_grid_EigenvaluesOnly(self.GetHoppings(),self.nbands,self.Kpoints_IBZ,self.lattice_vectors)
                else:
                    self.Eigenvalues,self.EigenVectors = Load_Hamiltonian_grid(self.GetHoppings(),self.nbands,self.Kpoints_IBZ,self.lattice_vectors)
            else:
                if self.EigenvaluesOnly == True:
                    self.Eigenvalues = Load_Hamiltonian_grid_EigenvaluesOnly(self.GetHoppings(),self.nbands,self.Kpoints_BZ,self.lattice_vectors)
                else:
                    self.Eigenvalues,self.EigenVectors = Load_Hamiltonian_grid(self.GetHoppings(),self.nbands,self.Kpoints_BZ,self.lattice_vectors)
                #self.Kpoints_BZ = Load_Hamiltonian_grid(self.GetHoppings(),self.nbands,nkx,nky,nkz,self.lattice_vectors,self.reciprocal_vectors)
            self.nkx = nkx
            self.nky = nky
            self.nkz = nkz

    def checkHermitian(self,kx,ky,kz):
        """
        Checks the Hamiltonian is Hermitian at a given k-point.
        """
        H = Load_Hamiltonian(self.GetHoppings(),kx,ky,kz,self.nbands,self.lattice_vectors)
        if np.allclose(H, H.T.conj()):
            return True
        else:
            print("NOT HERMITIAN!!!", kx,ky,kz)
            exit(0)

    def CalcDOS(self,omega,Gamma):
        """
        Calculate the density of states at a given energy omega and energy broadening term Gamma. 
        Calls the numba-ed function CalcDOS. 
        """
        dosinfo = [omega,CalcDOS(omega,Gamma,self.Eigenvalues,self.EigenVectors,self.Superconducting),Gamma] 
        self.DOS.append(dosinfo)
        if omega == 0.0:
            self.DOSatEF = dosinfo[1]
        return dosinfo[1]
    
    def CalcDOS_Orbital_resolved(self,omega,Gamma):
        """
        Calculate the density of states at a given energy omega and energy broadening term Gamma, but this time orbital resolved. 
        Calls the numba-ed function CalcDOS_Orbital_resolved. 
        returns a list of the density of states for each orbital specified using the DefineOrbitals function..
        """
        dos = CalcDOS_Orbital_resolved(omega,Gamma,self.Eigenvalues,self.EigenVectors,self.Superconducting)

        if self.Orbital_Repetition != 0:
            DOS = [0 for i in range(len(self.Orbital_Colour)+1)]
            for i in range(len(dos)-1):
                DOS[i%len(self.Orbital_Colour)] += dos[i]
            DOS[-1] = dos[-1]
            return DOS  
        else:
            return dos
    
    def CalcSommerfeldCoefficient(self,Gamma):
        eVtoJ = 1.60218e-19 #eV to J
        kb = 1.380e-23 #Boltzmann constant J/K
        AvagadrosNo = 6.022e23 #AvagadrosNo atoms/mol
        
        if self.DOSatEF == None:
            self.CalcDOS(0.0,Gamma) #calculate self.DOSatEF
            print(self.DOSatEF)
            sommerfeld_coeff = self.DOSatEF*(1000/eVtoJ)*AvagadrosNo*(kb**2)*(np.pi**2)/3.0
            print("Sommerfeld coefficient = ",sommerfeld_coeff,"mJ mol-1 K-2")
            return sommerfeld_coeff

    def CalcSpecificHeat(self,Temperature,Superconducting = False):
        if self.Kpoints_BZ is None:
            print("No k-grid loaded. Please load a k-grid using tbHamiltonian.Load_kgrid(nkx,nky,nkz)")
            exit(0)
        if self.symmetry == True:
            Kpoints = self.Kpoints_IBZ
        else:
            Kpoints = self.Kpoints_BZ
        #Ham_inst.ModifySuperconductingGap(Ham_inst.Tc+1) #set the gap to zero.
        if self.Superconducting == True:
            Cv_sc = self.CalcSpecificHeat_Superconducting(Kpoints,Temperature,Superconducting=Superconducting)
            return Cv_sc
        else:
            Cv = self.CalcSpecificHeat_NormalState(Kpoints,Temperature)
            return Cv
    
    def CalcSpecificHeat_Superconducting(self,Kpoints,Temperature,Superconducting=False):
        if Superconducting == True:
            h = 0.00001 #This is the small value required for numerical differentiation. 
            #Need to caculate the Hamiltonian for Delta(T), and it's temperature derivative. 
            self.ModifySuperconductingGap(Temperature)
            Hoppings_T = self.GetHoppings()
            self.ModifySuperconductingGap(Temperature+h)
            Hoppings_Tplus = self.GetHoppings()
            self.ModifySuperconductingGap(Temperature-h)
            Hoppings_Tminus = self.GetHoppings()
            #numba-ed function. 
            Cv = CalculateSpecificHeat_Superconducting(Kpoints,Hoppings_T,Hoppings_Tplus,Hoppings_Tminus,self.nbands,self.lattice_vectors,Temperature,self.Tc,h,self.SpinPolarised)
            return Cv
        else:
            self.ModifySuperconductingGap(self.Tc+1)
            Hoppings_T = self.GetHoppings()
            Cv = CalculateSpecificHeat_Superconducting_Normalstate(Kpoints,Hoppings_T,self.nbands,self.lattice_vectors,Temperature)
            return Cv

        

    def CalcSpecificHeat_NormalState(self,Kpoints,Temperature):
        Hoppings_T = self.GetHoppings() 

        Cv = CalculateSpecificHeat_NormalState(Kpoints,Hoppings_T,self.nbands,self.lattice_vectors,Temperature)
        return Cv
        
    #Tc = 1K. 
    #@timeit
    def ModifySuperconductingGap(self,T):
        """
        Given a superconducting hamiltonian, take the off diagonal block, and multiply it by a BCS temperature dependent factor. (see ModifyGap function).
        self.Tc should already be set, if not it will be assumed to be 1K. 
        """
        self.CurrentTemperature = T
        #TempGapParams.append([0.,0.,0.,Ham_index_1,Ham_index_2+int(self.nbands/2.0),L_updown.real,L_updown.imag,1.0]) 
        #TempGapParams.append([0.,0.,0.,Ham_index_1+int(self.nbands/2.0),Ham_index_2,L_downup.real,L_downup.imag,1.0]) 
        #print(self.nbands)

        self.GapRemoval = []
        if self.Superconducting == True:
            SCGap = self.ModifyGap(T/self.Tc) #between 0 and 1
            #SCGap = self.SuperconductingGap(T)
            for Hopping in self.hopping_parameters:
                #print(Hopping,SCGap)
                if (Hopping[3] > int(self.nbands/2.0)) and (Hopping[4] <= int(self.nbands/2.0)):
                    #print(Hopping[3],Hopping[4])
                    self.GapRemoval.append([Hopping[0],Hopping[1],Hopping[2],Hopping[3],Hopping[4],-(1-SCGap)*Hopping[5],-(1-SCGap)*Hopping[6],Hopping[7]])
                if (Hopping[3] <= int(self.nbands/2.0)) and (Hopping[4] > int(self.nbands/2.0)):
                    self.GapRemoval.append([Hopping[0],Hopping[1],Hopping[2],Hopping[3],Hopping[4],-(1-SCGap)*Hopping[5],-(1-SCGap)*Hopping[6],Hopping[7]])

    def ModifyGap(self,T):
        "Calculates the BCS temperature dependent gap. "
        if 0 <np.abs(T) < 1:
            #print(T)
            return np.tanh(1.76*np.sqrt((1.0)/np.abs(T) - 1)) 
        elif T == 0:
            return 1
        else:
            return 0
    
    #Put temperature in kelvin
    def CalcParticleNumber(self,omega,Temperature_kelvin):
        #kb = 8.617333262145E-5
        T_eV = Temperature_kelvin**8.617333262145E-5
        if self.Eigenvalues is not None:
            return CalcParticleNumber(omega,self.Eigenvalues,T_eV)
        else:
            print("No k-grid loaded. Please load a k-grid using tbHamiltonian.Load_kgrid(nkx,nky,nkz)")
            return 0
    
        #Put temperature in kelvin
    def CalcParticleNumber_Orbital_resolved(self,omega,Temperature_kelvin):
        #kb = 8.617333262145E-5
        T_eV = Temperature_kelvin**8.617333262145E-5
        if self.Eigenvalues is not None:
            N = CalcParticleNumber_Orbital_resolved(omega,self.Eigenvalues,self.EigenVectors,T_eV)
            ParticleNumbers = [0 for i in range(len(self.Orbital_Colour)+1)] # add up the 
            for i in range(len(N)-2):
                ParticleNumbers[i%len(self.Orbital_Colour)] += N[i]
            ParticleNumbers[-1] = N[-2]
            #print("Total_numberOfElectrons",N[-1])
            return ParticleNumbers
        else:
            print("No k-grid loaded. Please load a k-grid using tbHamiltonian.Load_kgrid(nkx,nky,nkz)")
            return 0

    #Takes a k point defined in units of k1*b1 k2*b2 and k3*b3, and converts it to the orthogonal kx*X,ky*Y,kz*Z co-ordinates. 
    def Convert_Kspace_To_XYZ(self,k1,k2,k3):
        kx = k1*self.reciprocal_vectors[0][0] +k2*self.reciprocal_vectors[1][0] + k3*self.reciprocal_vectors[2][0]
        ky = k1*self.reciprocal_vectors[0][1] +k2*self.reciprocal_vectors[1][1] + k3*self.reciprocal_vectors[2][1]
        kz = k1*self.reciprocal_vectors[0][2] +k2*self.reciprocal_vectors[1][2] + k3*self.reciprocal_vectors[2][2]
        return kx,ky,kz
    
    def Calc2DFermiSurfaceContour(self,N_BrillouinZone=1):
        edges = MarchingCube2D(self.Eigenvalues,isolevel=0.0,n_repetition=(N_BrillouinZone*2 -1)) #repetition defines how far beyond the 1st BZ we go. If it's 2 that means we sample 3 BZ (-1,0,+1).
        edge_count = 0 #Number of lines can be different for each FS plot. need to count in advance. 
        for i in range(len(edges)):
            edge_count+=len(edges[i])
        #print(edge_count)
        #exit(0)
        segs = np.zeros((edge_count, 2, 2))
        Colours = ["" for i in range(edge_count)]
        #nlines = len(EigVal)*len(EigenStore)
        #Loop over every pair of joining eigenvalues and make an (x1,y1) and (x2,y2) pair for each. ([segs[i][0][0],segs[i][0][1]]) and ([segs[i][1][0],segs[i][1][1]])
        line_counter =0
        for band_edge_index,band_edge in enumerate(edges):
            for edge in band_edge:
                #These four get the correct shape. edges are just indices of k points (e.g a 16x16 kpoint grid will have coordinates between [0,0] and [15,15]).
                #We have to first get that to between [0,0] and [1,1] and then we need to multiply them to get them in the range of [-b1,-b2], [+b1,+b2] where b1 and b2 are the reciprocal lattice vectors.
                # we thus do kx = p1/nkx * b1x + p2/nky * b1y  and kxy = p1/nkx * b2x + p2/nky * b2y for each ky pair. 
                #for FermiSurface, we only need kx,ky, but the function returns kx,ky,kz. Hence we save the kz to dummy_ and never use it. 
                segs[line_counter][0][0],segs[line_counter][0][1],dummy_ = self.Convert_Kspace_To_XYZ(edge[0][0]/(self.nkx-1),edge[0][1]/(self.nky-1),0.0) #Start #(edge[0][0]/Ham_inst.nkx)*Ham_inst.reciprocal_vectors[0][0] + (edge[0][1]/Ham_inst.nky)*Ham_inst.reciprocal_vectors[1][0]
                segs[line_counter][1][0],segs[line_counter][1][1],dummy_ = self.Convert_Kspace_To_XYZ(edge[1][0]/(self.nkx-1),edge[1][1]/(self.nky-1),0.0) #End 
                
                #This bit just shifts the Fermi surface to the center of the plot (i.e not [0,0] to [1,1] but [-0.5,-0.5] to [0.5,0.5])
                shiftocenter = (0.5 +(N_BrillouinZone-1))
                shiftocenter_kx,shiftocenter_ky,dummy_ = self.Convert_Kspace_To_XYZ(shiftocenter,shiftocenter,0.0)
                #(shiftocenter*Ham_inst.reciprocal_vectors[0][0] + shiftocenter*Ham_inst.reciprocal_vectors[1][0])
                segs[line_counter][0][0] -= shiftocenter_kx
                segs[line_counter][0][1] -= shiftocenter_ky

                segs[line_counter][1][0] -= shiftocenter_kx
                segs[line_counter][1][1] -= shiftocenter_ky

        return segs #segs is a list of lines, e.g segs[line0] = [[x1,y1],[x2,y2]]

@nb.njit
def CalculateSpecificHeat_Superconducting(Kpoints,hoppings_T,Hoppings_Tplus,Hoppings_Tminus,nbands,lattice_vectors,Temperature,Tc,increment,SpinPolarised):
    C = 0
    EdEdT = 0
    for k in Kpoints:
        Ham_sc = Load_Hamiltonian(hoppings_T,k[0],k[1],0,nbands,lattice_vectors)
        Ham_nsc = Ham_sc[:int(nbands/2),:int(nbands/2)]
        eigval_nsc,_ = LA.eigh(Ham_nsc) #Normal state eigenvalues ek
        #print(eigval_nsc,len(eigval_nsc))

        if Temperature <= Tc:
            Delta_k = GetSCGap(hoppings_T,nbands,lattice_vectors,k[0],k[1],0,SpinPolarised=SpinPolarised)
            Delta_k_Tplus = GetSCGap(Hoppings_Tplus,nbands,lattice_vectors,k[0],k[1],0,SpinPolarised=SpinPolarised)
            Delta_k_Tminus = GetSCGap(Hoppings_Tminus,nbands,lattice_vectors,k[0],k[1],0,SpinPolarised=SpinPolarised)
        else:
            Delta_k = np.array([0 for i in range(len(eigval_nsc))],dtype=np.float64)
            Delta_k_Tplus = np.array([0 for i in range(len(eigval_nsc))],dtype=np.float64)
            Delta_k_Tminus = np.array([0 for i in range(len(eigval_nsc))],dtype=np.float64)

        for band in range(len(eigval_nsc)):
            E_k_SC = np.sqrt(float(eigval_nsc[band])**2 + float(Delta_k[band]**2)) #E_k_SC = sqrt(ek^2 + Delta_k^2)
            #print(E_k_SC,eigval_nsc[band],Delta_k[band])

            F = -FermiFunction_derivative_dE(np.real(E_k_SC),Temperature)
            C += F*(E_k_SC**2)/Temperature 
            if Temperature < Tc:
                E_k_SC_plus = np.sqrt(float(eigval_nsc[band])**2 + float(Delta_k_Tplus[band]**2)) #E_k_SC = sqrt(ek^2 + Delta_k^2)
                E_k_SC_minus = np.sqrt(float(eigval_nsc[band])**2 + float(Delta_k_Tminus[band]**2)) #E_k_SC = sqrt(ek^2 + Delta_k^2)

                EdEdT += F*E_k_SC*((E_k_SC_plus-E_k_SC_minus)/(2*increment))

    print(Temperature,(C/len(Kpoints)),(EdEdT/len(Kpoints)),(C-EdEdT)/len(Kpoints),(C-EdEdT)/(len(Kpoints)*Temperature))
    return (C-EdEdT)/len(Kpoints) # C is normal state specific heat, EdEdT is the superconducting part. It is zero if not superconducting.


def CalculateSpecificHeat_Superconducting_Normalstate(Kpoints,hoppings_T,nbands,lattice_vectors,Temperature):
    C = 0
    EdEdT = 0
    for k in Kpoints:
        Ham_sc = Load_Hamiltonian(hoppings_T,k[0],k[1],0,nbands,lattice_vectors)
        Ham_nsc = Ham_sc[:int(nbands/2),:int(nbands/2)]
        eigval_nsc,_ = LA.eigh(Ham_nsc) #Normal state eigenvalues ek

        for band in range(len(eigval_nsc)):
            F = -FermiFunction_derivative_dE(np.real(eigval_nsc[band]),Temperature)
            C += F*(eigval_nsc[band]**2)/Temperature 
    #print(Temperature,(C/len(Kpoints)),(EdEdT/len(Kpoints)),(C-EdEdT)/len(Kpoints),(C-EdEdT)/(len(Kpoints)*Temperature))
    return (C)/len(Kpoints) # C is normal state specific heat, EdEdT is the superconducting part. It is zero if not superconducting.

@nb.njit
def CalculateSpecificHeat_NormalState(Kpoints,hoppings_T,nbands,lattice_vectors,Temperature):
    C = 0
    for k in Kpoints:
        Ham = Load_Hamiltonian(hoppings_T,k[0],k[1],0,nbands,lattice_vectors)
        eigval,_ = LA.eigh(Ham) #Normal state eigenvalues ek
        for band in range(len(eigval)):
            F = -FermiFunction_derivative_dE(np.real(eigval[band]),Temperature)
            C += F*(eigval[band]**2)/Temperature 

    #print(Temperature,(C/len(Kpoints)),(EdEdT/len(Kpoints)),(C-EdEdT)/len(Kpoints),(C-EdEdT)/(len(Kpoints)*Temperature))
    return (C)/len(Kpoints) # C is normal state specific heat, EdEdT is the superconducting part. It is zero if not superconducting.

@nb.njit
def calcOrbitalWeights(eigenvalue_index,eigenvector):
    Orbital_weights = np.zeros(len(eigenvector),dtype=np.float64)
    for i in range(len(eigenvector)): #every orbital index
        Orbital_weights[i] += np.round(abs(eigenvector[i][eigenvalue_index])**2,6)
    return Orbital_weights

kb_eV = 8.617333262145E-5
@nb.njit
def FermiFunction(Energy, Temperature):
    if Temperature == 0.0:
        if Energy < 0.0:
            return 1.0
        else:
            return 0.0
    else:
        return 1.0/(np.exp(Energy/(Temperature*kb_eV))+1.0)

@nb.njit
def FermiFunction_derivative_dE(Energy, Temperature):
        kbT = kb_eV*Temperature
        exp = np.exp(Energy/kbT)
        if exp > 1E10:
            return 0.0
        
        Fd = -(1.0/kbT)*exp/((exp+1)**2)
        if (np.isnan(Fd) == True):
            #print("yep that's a nan")
            return 0.0
        else:
            return Fd

        
@nb.njit
def FermiFunction_derivative_dT(Energy, Temperature):
        kbT = kb_eV*Temperature
        exp = np.exp(Energy/kbT)
        if exp > 1E10:
            return 0.0
        
        Fd = (Energy/(kbT**2))*exp/((exp+1)**2)
        if Fd == np.nan:
            return 0.0
        else:
            return Fd
        
@nb.njit
def GetSCGap(hoppings,nbands,lattice_vectors,kx,ky,kz,SpinPolarised=False):
    #print(kx,ky,kz)
    """
     Given a superconducting hamiltonian, extract the gap on each band at a specific k point.
     The idea is that we calculate the non superconducting eigenvalues (nsc)from the first block. Then the superconducting eigenvalues (sc) from the full matrix.
     For each non-superconducting eigenvalue, we then find the closest superconducting eigenvalue. Which should be E_sc = sqrt(E_nsc^2 + Delta^2). Then we rearrange this to find the Delta, which is the gap.

     returns: magnitude of superconducting gap (Delta) for each band at a given k.
    """
    H = Load_Hamiltonian(hoppings,kx ,ky,kz,nbands,lattice_vectors)
    eigval_nsc,eigvec_nsc = LA.eigh(H[:int(nbands/2),:int(nbands/2)])
    eigval_sc,eigvec_sc = LA.eigh(H)
    Delta = []


    #Need to calculate <Ui|CkCk|Ui> for each band.
    #CkCk is a matrix of zeros, with a 1 in the off diagonal block that corresponds to the superconducting gap (singlet in this case)
    # if your hamiltonian is not spin polarised 
    # |H(k), Delta(k) |
    # |Delta(k),-H(-k)|    
    #CkCk =
    # |0, 1|
    # |0, 0|
    # else, if your hamiltonian is spin polarised
    # #H =
    # |H(k)+Lz,Lx+iLy,                       Delta(k)^UpUp, Delta(k)^UpDown   |
    # |Lx-iLy,H(k)-Lz,                       Delta(k)^DownUp,Delta(k)^DownDown|   
    # |Delta(k)^UpUp, Delta(k)^UpDown,      -H(-k)+Lz,Lx+iLy,                 |
    # |Delta(k)^DownUp,Delta(k)^DownDown ,   Lx-iLy,-H(-k)-Lz,                |  

    #CkCk (singlet) = 
    # |0, 0 0 1|
    # |0, 0 1 0|
    # |0, 0 0 0|
    # |0, 0 0 0|

    #CkCk (triplet) = 
    # |0, 0 1 0|
    # |0, 0 0 1|
    # |0, 0 0 0|
    # |0, 0 0 0|


    ckck = np.zeros((nbands,nbands),dtype=np.complex128)
    #ckck[0][1] = 1
    half_nbands = int(nbands/2)
    quart_nbands = int(nbands/4)
    if SpinPolarised == True:
        for i in range(quart_nbands):
            ckck[i][i+half_nbands+quart_nbands] = -1
            ckck[i+int(nbands/4)][i+half_nbands] = 1
            ckck[i+half_nbands+quart_nbands][i] = -1
            ckck[i+half_nbands][i+int(nbands/4)] = 1
    else:
        for i in range(half_nbands):
            ckck[i][i+half_nbands] = 1

    #
    orbitalDirector = np.zeros(nbands,dtype=np.complex128)
    for i in range(half_nbands): #choose spin up orbitals if spin polarised.
        orbitalDirector[i] = 1

    for E_nsc in eigval_nsc:
        absolute_diff = np.abs(eigval_sc - E_nsc) 
        # Find the index of the element with the minimum difference
        #print(absolute_diff)

        closest_index = 0#np.argmin(absolute_diff)
        if np.abs(eigval_sc[closest_index] - eigval_sc[closest_index+1]) < 1e-5:
            index_1 = np.real(np.abs(np.dot(np.conj(eigvec_sc[:,closest_index]),orbitalDirector)))
            index_2 = np.real(np.abs(np.dot(np.conj(eigvec_sc[:,closest_index+1]),orbitalDirector)))
            #print(index_1,index_2)
            if index_2>index_1:
                closest_index = closest_index+1 
        """
        closest_index = np.argmin(absolute_diff)
        if closest_index != 0:
            if np.abs(eigval_sc[closest_index-1] - eigval_sc[closest_index]) < 1e-5:
                closest_index = closest_index-1
    
        
        if np.abs(eigval_sc[closest_index] - eigval_sc[closest_index+1]) < 1e-5:
            index_1 = np.real(np.abs(np.dot(np.conj(eigvec_sc[:,closest_index]),orbitalDirector)))
            index_2 = np.real(np.abs(np.dot(np.conj(eigvec_sc[:,closest_index+1]),orbitalDirector)))
            #print(index_1,index_2)
            if index_2>index_1:
                closest_index = closest_index+1 
        """
        
        # Return the closest element and its index
        E_SC = eigval_sc[closest_index] #

        eigenvector = np.zeros(nbands,dtype=np.complex128)
        for i in range(nbands):
            eigenvector[i] = eigvec_sc[i][closest_index]
        #eigenvector = eigvec_sc[:,closest_index] #numba didn't like this...
        Mat = np.dot(ckck,eigenvector)
        newMat = np.dot(np.conj(eigenvector),Mat)
        sign = np.sign(np.real(newMat))

        Delta.append(sign*np.sqrt(np.abs(np.real(E_SC)**2 - np.real(E_nsc)**2)))  
    #print(Delta)
    #exit(0)
    return np.array(Delta,dtype=np.float64) #List of Delta for each band at a given k and Temperature. 



@nb.njit
def CalcParticleNumber(omega,Energy_store,Temperature):
    Emin,Emax = determinFermiFunctionBoundaries(Temperature)
    nk,nbands = np.shape(Energy_store)
    N = 0
    for k in range(len(Energy_store)):
        for band in range(nbands):
            if (Emin < np.real(Energy_store[k][band]) < Emax):
                #N += 1
                N += FermiFunction(np.real(Energy_store[k][band])-omega,Temperature)
    return N/(nk)

#Get working on Monday!
@nb.njit
def CalcParticleNumber_Orbital_resolved(omega,Energy_store,Eigenvectors_store,Temperature):
    Emin,Emax = determinFermiFunctionBoundaries(Temperature)
    nk,nbands = np.shape(Energy_store)
    N = np.zeros(nbands+2,dtype=np.float64)
    for k in range(len(Energy_store)):
        for band in range(nbands):
            if (Emin < np.real(Energy_store[k][band]) < Emax):
                #N += 1
                FermiFunc = FermiFunction(np.real(Energy_store[k][band])-omega,Temperature)
                for s in range(nbands): #orbital
                    N[s] += FermiFunc*np.real(Eigenvectors_store[k][s][band]*np.conj(Eigenvectors_store[k][s][band]))
                N[-2] += FermiFunc # This one adds up the total number of bands, same as CalcParticleNumber as above. 
                N[-1] += 1 #This one gets the total number of bands - not needed, but a good check. 
    for i in range(len(N)):
        N[i] = N[i]/(nk)
    return N

@nb.njit
def determinFermiFunctionBoundaries(T):
    Emin = 0
    Emax = 0
    for E in np.arange(0,-1000,-0.1):
        if FermiFunction(E,T) > 0.999999:
            Emin = E
            break
    for E in np.arange(0,1000,0.1):
        if FermiFunction(E,T) < 0.000001:
            Emax = E
            break
    return Emin,Emax


@nb.njit
def Load_Hamiltonian(HAM,kx,ky,kz,nbands,lattice_vectors):
    #H(k) = Sum_R t(Rx,Ry,Rz)*e^(i(kx*Rx + ky*Ry + kz*Rz)
    H = np.zeros((nbands, nbands), dtype=np.complex128)
    for i in range(len(HAM)):
        #print(HAM[i])
        #This converts from primative lattice vectors to Rx,Ry,Rz orthogonal co-ordinates. 
        Rx = HAM[i][0]*lattice_vectors[0][0] + HAM[i][1]*lattice_vectors[1][0] + HAM[i][2]*lattice_vectors[2][0]
        Ry = HAM[i][0]*lattice_vectors[0][1] + HAM[i][1]*lattice_vectors[1][1] + HAM[i][2]*lattice_vectors[2][1]
        Rz = HAM[i][0]*lattice_vectors[0][2] + HAM[i][1]*lattice_vectors[1][2] + HAM[i][2]*lattice_vectors[2][2]
        #print(HAM[i])
        #print(np.exp(1j * (kx * Rx + ky * Ry+ kz * Rz)),HAM[i][5],1j*HAM[i][6],HAM[i][7])
        H[int(HAM[i][3])-1][int(HAM[i][4])-1] += (HAM[i][5] +1j*HAM[i][6])* np.exp(1j * (kx * Rx + ky * Ry+ kz * Rz))/HAM[i][7]
    return H

@timeit
@nb.njit
def Load_Hamiltonian_grid(HAM,nbands,KPOINTS,lattice_vectors):
    Energy_store = np.zeros((len(KPOINTS),nbands),dtype=np.complex128)
    Evectors_store = np.zeros((len(KPOINTS),nbands,nbands),dtype=np.complex128)
    for Ki,K in enumerate(KPOINTS):
        H = Load_Hamiltonian(HAM,K[0],K[1],K[2],nbands,lattice_vectors)
        Eigenvalues,Eigenvectors = LA.eigh(H)
        Energy_store[Ki] = Eigenvalues 
        Evectors_store[Ki] = Eigenvectors
    return Energy_store,Evectors_store


#Only use this if the hamiltonian is a superconductor and you want to get the non-superconducting eigenvalues!
#Calculates the first block of a superconducting (Nambau) hamiltonian. 
@timeit
#@nb.njit
def Load_Hamiltonian_grid_NormalState(HAM,nbands,KPOINTS,lattice_vectors):
    Energy_store = np.zeros((len(KPOINTS),int(nbands/2)),dtype=np.complex128)
    Evectors_store = np.zeros((len(KPOINTS),int(nbands/2),int(nbands/2)),dtype=np.complex128)
    for Ki,K in enumerate(KPOINTS):
        H = Load_Hamiltonian(HAM,K[0],K[1],K[2],nbands,lattice_vectors)
        Eigenvalues,Eigenvectors = LA.eigh(H[:int(nbands/2),:int(nbands/2)])
        Energy_store[Ki] = Eigenvalues 
        Evectors_store[Ki] = Eigenvectors
    return Energy_store,Evectors_store

@timeit
@nb.njit
def Load_Hamiltonian_grid_EigenvaluesOnly(HAM,nbands,KPOINTS,lattice_vectors):
    Energy_store = np.zeros((len(KPOINTS),nbands),dtype=np.complex128)
    #Evectors_store = np.zeros((len(KPOINTS),nbands,nbands),dtype=np.complex128)
    for Ki,K in enumerate(KPOINTS):
        H = Load_Hamiltonian(HAM,K[0],K[1],K[2],nbands,lattice_vectors)
        Eigenvalues,Eigenvectors = LA.eigh(H)
        Energy_store[Ki] = Eigenvalues
        #Evectors_store[Ki] = Eigenvectors
    return Energy_store#,Evectors_store

@timeit
@nb.njit
def Load_Kpoints_BZ(nkx,nky,nkz,reciprocal_vectors):
    #Kpoints_Store = []
    Kpoints_Store = np.zeros((nkx*nky*nkz,3),dtype=np.float64)
    counter = 0
    for i in range(nkx):
            xi = (2.0*(i+0.5)-nkx)/(2.0*nkx)
            for j in range(nky):
                yi = (2.0*(j+0.5)-nky)/(2.0*nky)
                for k in range(nkz):
                    if (nkz==1):zi=0.0
                    else:zi = (2.0*(k+0.5)-nkz)/(2.0*nkz)
                    kx = xi*reciprocal_vectors[0][0] + yi*reciprocal_vectors[1][0] + zi*reciprocal_vectors[2][0]
                    ky = xi*reciprocal_vectors[0][1] + yi*reciprocal_vectors[1][1] + zi*reciprocal_vectors[2][1]
                    kz = xi*reciprocal_vectors[0][2] + yi*reciprocal_vectors[1][2] + zi*reciprocal_vectors[2][2]
                    
                    #Kpoints_Store.append([np.round(kx,6),np.round(ky,6),np.round(kz,6)])
                    Kpoints_Store[counter][0],Kpoints_Store[counter][1],Kpoints_Store[counter][2] = np.round(kx,6),np.round(ky,6),np.round(kz,6)
                    counter+=1
    return Kpoints_Store

class tbHamiltonian_Plot():
    r'''
    Plots useful model bits.
    #Press the keyboard arrow keys (UP and DOWN) to change the bias/energy of the image.

    Args:


    Attributes:

    Methods:
        PlotDOS(Emin,Emax,Nsteps,Gamma)
            Plots the density of states.
        BandStructure(kpath,klabels,YMIN,YMAX,nk,color)
            Plots the band structure.
        FermiSurface2D(omega,kz,color)
            Plots the 2D Fermi surface at a given energy and kz.


    '''

    def __init__(self, tbHamiltonian):
        #If you only provide a single tbHamiltonian, make it a list, and then iterate over it.
        #otherwise iterate over the list of tbHamiltonians.
        self.tbHamiltonian=[]
        self.tbHamiltonianFileNames= []
        try:
            Ham_iterator = iter(tbHamiltonian)
        except TypeError:
            Ham_iterator = iter([tbHamiltonian])
        for idx, Ham_inst in enumerate(Ham_iterator):
            self.tbHamiltonian.append(Ham_inst)
            if (self.tbHamiltonian[idx].nkx == 0) or (self.tbHamiltonian[idx].nky == 0) or (self.tbHamiltonian[idx].nkz==0):
                if self.tbHamiltonian[idx].no_kgrid == False:
                    print("No k-grid loaded. Please load a k-grid using tbHamiltonian.Load_kgrid(nkx,nky,nkz) or specify no_kgrid=True in tbHamiltonian()")
                    print("going to use automatic kgrid of 128x128x1")
                    self.tbHamiltonian[idx].Load_kgrid(128,128,1)
        
        #self.nbands = tbHamiltonian.nbands

    def PlotDOS(self,E_start,E_end,Nsteps,Gamma,color="#680C07"):
        """
        E_start = energy start value (eV)
        E_end = energy end value (eV)
        Nsteps = number of steps between E_start and E_end. 
        Gamma = energy broadening (eV)
        """
        if isinstance(color, str):
            color = [color]
        self.fig_DOS, self.ax_DOS = plt.subplots()
        x = np.linspace(E_start,E_end,Nsteps)
        maxy = 0
        miny = 0
        DOS_list = []
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            y = []
            for E in x:
                DOS = Ham_inst.CalcDOS(E,Gamma)
                if DOS > maxy: maxy = DOS
                if DOS < miny: miny = DOS
                y.append(DOS)
            self.ax_DOS.plot(x,y,color=color[idx%len(color)])
            DOS_list.append(y)
           
        YMIN = miny
        YMAX = maxy*1.05
        if (E_end/E_start) < 0:
            self.ax_DOS.vlines(x=0,ymin=YMIN,ymax=YMAX,lw=1,color="grey")
        
        self.ax_DOS.set_xlim([E_start,E_end])
        self.ax_DOS.set_ylim([YMIN,YMAX])
        return x,DOS_list  
        #plt.show()
    
    def PlotDOS_orbital_resolved(self,E_start,E_end,Nsteps,Gamma,showLegend=True):
        """
        E_start = energy start value (eV)
        E_end = energy end value (eV)
        Nsteps = number of steps between E_start and E_end. 
        Gamma = energy broadening (eV)
        """
        self.fig_DOS, self.ax_DOS = plt.subplots()
        x = np.linspace(E_start,E_end,Nsteps)

        maxy = 0
        miny = 0
        DOS_list = []
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            y = [[] for i in range(len(Ham_inst.Orbital_Colour)+1)]
            for E in x:
                DOS = Ham_inst.CalcDOS_Orbital_resolved(E,Gamma)
                if DOS[-1] > maxy: maxy = DOS[-1]
                if DOS[-1] < miny: miny = DOS[-1]
                for orb in range(len(DOS)):
                    y[orb].append(DOS[orb])
            Colours = Ham_inst.Orbital_Colour[:]
            Colours.append("black")
            Labels = Ham_inst.Orbital_Label[:]
            Labels.append("Total")
            for i in range(len(y)): 
                self.ax_DOS.plot(x,y[i],c=Colours[i],label=Labels[i])
            DOS_list.append(y)
            
        if showLegend == True: plt.legend()
           
        YMIN = miny
        YMAX = maxy*1.05
        if (E_end/E_start) < 0:
            self.ax_DOS.vlines(x=0,ymin=YMIN,ymax=YMAX,lw=1,color="grey")
        
        self.ax_DOS.set_xlim([E_start,E_end])
        self.ax_DOS.set_ylim([YMIN,YMAX])
        return x,DOS_list  
        #plt.show()

    def BandStructure(self,kpath=[[0,0,0],[np.pi,np.pi,0]],klabels=[r"$\Gamma$","M"],YMIN=-1,YMAX=+1,nk=50,color="#680C07",linewidth=2,alpha=1.0,Use_KPath_Distance=True):
        #self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(111)  
        self.fig_BandStructure, self.ax_BandStructure = plt.subplots()  
        count = 0
        # if color is a string, turn it into a list of strings
        if isinstance(color, str):
            color = [color]

        if not isinstance(linewidth, list): #If linewidths is a single value, turn it into a list of values.
            linewidth = [linewidth]
        if not isinstance(alpha, list): #If alphas is a single value, turn it into a list of values.
            alpha = [alpha]
        
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            kdistance_store = []
            EigenStore = []
            Orbital_colour_store = []
            Total_Kdistance = 0
            Kdistance = 0
            kdistance_HighSymmetryPoints = []
            for i in range(len(kpath)-1):
                Xstart,Ystart,Zstart = Ham_inst.Convert_Kspace_To_XYZ(kpath[i][0],kpath[i][1],kpath[i][2])
                Xend,Yend,Zend = Ham_inst.Convert_Kspace_To_XYZ(kpath[i+1][0],kpath[i+1][1],kpath[i+1][2])#  kpath[i][0]*Ham_inst.reciprocal_vectors[0][0] + kpath[i][1]*Ham_inst.reciprocal_vectors[1][0] + kpath[i][2]*Ham_inst.reciprocal_vectors[2][0]
                Xdiff,Ydiff,Zdiff = (Xend-Xstart),(Yend-Ystart),(Zend-Zstart)
                
                Total_Kdistance +=Kdistance*nk #Total distance in k-space travelled. 
                kdistance_HighSymmetryPoints.append(Total_Kdistance) #This is the value where we plot our k-labels. 
                if Use_KPath_Distance==True:
                    Kdistance = np.sqrt((Xdiff/nk)**2 +(Ydiff/nk)**2 + (Zdiff/nk)**2)
                else:
                    Kdistance = 1

                #get the Eigenvalues, and k-distance for each k point 
                for x in range(nk):
                    xi =  Xstart + Xdiff*(x/nk)
                    yi = Ystart + Ydiff*(x/nk)
                    zi = Zstart + Zdiff*(x/nk)
                    H = Load_Hamiltonian(Ham_inst.GetHoppings(),xi,yi,zi,Ham_inst.nbands,Ham_inst.lattice_vectors)
                    EigVal,EigVec = LA.eigh(H)
                    EigenStore.append(EigVal)
                    kdistance_store.append(Total_Kdistance + Kdistance*x )

            #x = [x for x in range(len(EigenStore))]
            self.ax_BandStructure.plot(kdistance_store,EigenStore,color=color[idx%len(color)],linewidth=linewidth[idx%len(linewidth)],alpha=alpha[idx%len(alpha)])

        Total_Kdistance +=Kdistance*nk
        kdistance_HighSymmetryPoints.append(Total_Kdistance) #Add in the final high symmetry point
        for i in range(len(kdistance_HighSymmetryPoints)):
            self.ax_BandStructure.vlines(x=kdistance_HighSymmetryPoints[i],ymin=YMIN, ymax=YMAX,lw=1,color="grey")
            #self.ax.vlines(x=nk*(i+1), ymin=YMIN, ymax=YMAX,lw=1,color="grey")

        self.ax_BandStructure.set_xlim([0,kdistance_store[-1]])
        self.ax_BandStructure.set_ylim([YMIN, YMAX])

        self.ax_BandStructure.set_xticks(kdistance_HighSymmetryPoints)
        self.ax_BandStructure.set_xticklabels(klabels)
        self.ax_BandStructure.set_ylabel("E (eV)")
        self.ax_BandStructure.set_xlabel("k")
        if (YMAX/YMIN) < 0: #sign change, zero exists
            self.ax_BandStructure.hlines(y=0,xmin=0,xmax=kdistance_store[-1],lw=1,color="grey")

        return kdistance_store,EigenStore
    
    def BandStructure_orbital_resolved(self,kpath=[[0,0,0],[np.pi,np.pi,0]],klabels=[r"$\Gamma$","M"],YMIN=-1,YMAX=+1,nk=50,color="#680C07",linewidth=2,alpha=1.0,Use_KPath_Distance=True):
        #Color does nothing here, but i include it so that the code doesn't break when you swap between orbital_resolved and non-orbital resolved versions. 
        self.fig_BandStructure_orbital, self.ax_BandStructure_orbital = plt.subplots()  

        count = 0

        if not isinstance(linewidth, list): #If linewidths is a single value, turn it into a list of values.
            linewidth = [linewidth]
        if not isinstance(alpha, list): #If alphas is a single value, turn it into a list of values.
            alpha = [alpha]
        
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            if len(Ham_inst.Orbital_Colour) <1:
                print(f"Error: No orbital colours specified for Hamiltonian {idx}. Please specify orbital colours using Ham.DefineOrbitals([orbital_label_1,orbital_label_2,...],[colour_1,colour_2,...])")
                return 0
            self.kdistance_store = []
            EigenStore = []
            Orbital_colour_store = []
            Total_Kdistance = 0
            Kdistance = 0
            kdistance_HighSymmetryPoints = []
            for i in range(len(kpath)-1):
                #Xstart,Ystart,Zstart = kpath[i][0],kpath[i][1],kpath[i][2]
                Xstart,Ystart,Zstart = Ham_inst.Convert_Kspace_To_XYZ(kpath[i][0],kpath[i][1],kpath[i][2])
                Xend,Yend,Zend = Ham_inst.Convert_Kspace_To_XYZ(kpath[i+1][0],kpath[i+1][1],kpath[i+1][2])#  kpath[i][0]*Ham_inst.reciprocal_vectors[0][0] + kpath[i][1]*Ham_inst.reciprocal_vectors[1][0] + kpath[i][2]*Ham_inst.reciprocal_vectors[2][0]
                Xdiff,Ydiff,Zdiff = (Xend-Xstart),(Yend-Ystart),(Zend-Zstart)
                #Xend,Yend,Zend = kpath[i+1][0],kpath[i+1][1],kpath[i+1][2]
                Total_Kdistance +=Kdistance*nk #Total distance in k-space travelled. 
                kdistance_HighSymmetryPoints.append(Total_Kdistance) #This is the value where we plot our k-labels. 
                if Use_KPath_Distance==True:
                    Kdistance = np.sqrt((Xdiff/nk)**2 +(Ydiff/nk)**2 + (Zdiff/nk)**2)
                else:
                    Kdistance = 1

                
                for x in range(nk+1):
                    xi =  Xstart + Xdiff*(x/nk)
                    yi = Ystart + Ydiff*(x/nk)
                    zi = Zstart + Zdiff*(x/nk)
                    H = Load_Hamiltonian(Ham_inst.GetHoppings(),xi,yi,zi,Ham_inst.nbands,Ham_inst.lattice_vectors)
                    EigVal,EigVec = LA.eigh(H)
                    EigenStore.append(EigVal)
                    
                    self.kdistance_store.append(Total_Kdistance + Kdistance*x )
                    Orbital_colour_store.append([])
                    for E in range(len(EigVal)):
                        if Ham_inst.MixOrbitalWeights == True: #Mix the weight, or just take the maximum 
                            Mix = np.sum([np.multiply(Ham_inst.GetOrbitalWeights(E,EigVec), Ham_inst.Orbital_Colour[i]) for i in range(len(Ham_inst.Orbital_Colour))],axis=1)
                            if len(Mix) !=3:
                                Mix = (Mix[0],0,0)
                            Orbital_colour_store[-1].append(Mix)
                        else:
                            orbWeights =  Ham_inst.GetOrbitalWeights(E,EigVec)
                            #print(np.argmax(orbWeights))
                            unique_elements, indices = np.unique(orbWeights, return_index=True)
                            if len(Ham_inst.Orbital_Colour) >1:
                                if np.abs((orbWeights[indices[-1]] - orbWeights[indices[-2]])) < 0.01: #If two indices are approximately the same, always choose the lower index for the colour.
                                    if indices[-1] < indices[-2]:
                                        Orbital_colour_store[-1].append(Ham_inst.Orbital_Colour[indices[-1]])
                                    else:
                                        Orbital_colour_store[-1].append(Ham_inst.Orbital_Colour[indices[-2]])
                                else:
                                    Orbital_colour_store[-1].append(Ham_inst.Orbital_Colour[indices[-1]])
                            else:
                                Orbital_colour_store[-1].append(Ham_inst.Orbital_Colour[indices[-1]])
                                
            #Okay this was tricky to get working quickly. you can use plt.scatter(x[i],Eigenvalue[i],c=Orbital_colour_store[i]) to plot the orbitals, but it gets really really slow for large number of points.
            #LineCollections are much quicker, but require modifying the data into lists of [(x1,y1) (x2,y2) (x3,y3)...] using an array of size (nlines,npointsperline,2). Each one of these lines can have it's own colour and plots quickly. 
            #I've just made every line have a start and end point, so that each line can have it's own colour. 
            Colours = [(0,0,0) for i in range(len(EigVal)*len(EigenStore))]
            segs = np.zeros((len(EigVal)*len(EigenStore), 2, 2))
            nlines = len(EigVal)*len(EigenStore)
            #Loop over every pair of joining eigenvalues and make an (x1,y1) and (x2,y2) pair for each. ([segs[i][0][0],segs[i][0][1]]) and ([segs[i][1][0],segs[i][1][1]])
            for i in range(len(EigenStore)-1):
                for j in range(len(EigVal)):
                    segs[i*len(EigVal) +j][0][0] = self.kdistance_store[i]
                    segs[i*len(EigVal) +j][0][1] = EigenStore[i][j]
                    segs[i*len(EigVal) +j][1][0] = self.kdistance_store[i+1]
                    segs[i*len(EigVal) +j][1][1] = EigenStore[i+1][j]
                    Colours[i*len(EigVal) +j] = Orbital_colour_store[i][j]
            #The final point.
            for j in range(len(EigVal)):
                segs[(len(EigenStore)-1)*len(EigVal) +j][0][0] = (self.kdistance_store[-2])
                segs[(len(EigenStore)-1)*len(EigVal) +j][0][1] = EigenStore[-2][j]
                segs[(len(EigenStore)-1)*len(EigVal) +j][1][0] = (self.kdistance_store[-1])
                segs[(len(EigenStore)-1)*len(EigVal) +j][1][1] = EigenStore[-1][j]
                Colours[(len(EigenStore)-1)*len(EigVal) +j] = Orbital_colour_store[-1][j]

            #This is the magic function that makes plotting large number of lines with different colours so much quicker. 
            line_segments = LineCollection(segs, linewidths=4,colors=Colours,linewidth=linewidth[idx%len(linewidth)],alpha=alpha[idx%len(alpha)],linestyle='solid')
            self.ax_BandStructure_orbital.add_collection(line_segments)

        Total_Kdistance +=Kdistance*nk
        kdistance_HighSymmetryPoints.append(Total_Kdistance) #Add in the final high symmetry point
        for i in range(len(kdistance_HighSymmetryPoints)):
            self.ax_BandStructure_orbital.vlines(x=kdistance_HighSymmetryPoints[i],ymin=YMIN, ymax=YMAX,lw=1,color="grey")
            #self.ax_BandStructure_orbital.vlines(x=nk*(i+1), ymin=YMIN, ymax=YMAX,lw=1,color="grey")

        self.ax_BandStructure_orbital.set_xlim([0,self.kdistance_store[-1]])
        self.ax_BandStructure_orbital.set_ylim([YMIN, YMAX])

        self.ax_BandStructure_orbital.set_xticks(kdistance_HighSymmetryPoints)
        self.ax_BandStructure_orbital.set_xticklabels(klabels)
        self.ax_BandStructure_orbital.set_ylabel("E (eV)")
        self.ax_BandStructure_orbital.set_xlabel("k")
        if (YMAX/YMIN) < 0: #sign change, zero exists
            self.ax_BandStructure_orbital.hlines(y=0,xmin=0,xmax=self.kdistance_store[-1],lw=1,color="grey")
    
    def FermiSurface2D(self, omega, kz=0,color=["Grey"],alpha=[1]):
        if not (isinstance(color, list)): #Color needs to be a list, if only one value is specified, i.e a string, make it into a list. 
            color = [color]
        if not (isinstance(alpha, list)): #Alpha needs to be a list, if only one value is specified, i.e a single value, make it into a list.
            alpha = [alpha]

        self.fig_FS,self.ax_FS = plt.subplots()
        self.ax_FS.set_aspect('equal', 'box')

 

        
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            Eigenvalues = np.array(Ham_inst.Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,Ham_inst.nbands))        #convert into a nice cubic x,y,z grid of eigenvalues. 
            kz_values = np.unique(Ham_inst.Kpoints_BZ[:,2])
            kz_index = np.argmin(np.abs(np.array(kz_values) - kz))
            print(f"kz_index:{kz_values[kz_index]} (index {kz_index})")
            kx_min = Ham_inst.reciprocal_vectors[0][0] + Ham_inst.reciprocal_vectors[1][0] + Ham_inst.reciprocal_vectors[2][0]
            ky_min = Ham_inst.reciprocal_vectors[0][1] + Ham_inst.reciprocal_vectors[1][1] + Ham_inst.reciprocal_vectors[2][1]
            #kz_min = -0.5*Ham_inst.reciprocal_vectors[0][2] + -0.5*Ham_inst.reciprocal_vectors[1][2] + -0.5*Ham_inst.reciprocal_vectors[2][2]
            x = np.linspace(-kx_min,kx_min, Ham_inst.nkx)
            y = np.linspace(-ky_min, ky_min, Ham_inst.nky)
            #xi = np.linspace(-Ham_inst.nkx/(2.0*Ham_inst.nkx),Ham_inst.nkx/(2.0*))#(2.0*i-nkx)/(2.0*nkx)
            X, Y = np.meshgrid(x, y)
            for N in range(Ham_inst.nbands):
                V = Eigenvalues[:, :, kz_index, N]    - omega
                Vmin = np.min(V)
                Vmax = np.max(V)
                #print(Vmin,Vmax)
                if Vmax/Vmin <0: #sign change.
                    self.ax_FS.contour(X, Y, np.real(V), [0], colors=color[idx%len(color)], alpha=alpha[idx%len(alpha)], linewidths=3.0)

        
        self.ax_FS.set_xlabel(r"$k_x$")
        self.ax_FS.set_ylabel(r"$k_y$")
        self.ax_FS.set_aspect('equal', 'box')

    def FermiSurface2D_orbital_resolved(self, omega,N_BrillouinZone=1, kz=0,color=["Grey"],alpha=[1],linewidth=[4],showBrillouinZone=True,setAntiAliased=True):
        if not (isinstance(alpha, list)): #Alpha needs to be a list, if only one value is specified, i.e a single value, make it into a list.
            alpha = [alpha]
        if not (isinstance(linewidth, list)): #Alpha needs to be a list, if only one value is specified, i.e a single value, make it into a list.
            linewidth = [linewidth]

        #color and alpha do nothing - They are for compatibility with the non-orbital resolved version.

        self.fig_FS_orbColour,self.ax_FS_orbColour = plt.subplots()
        self.ax_FS_orbColour.set_aspect('equal', 'box')
        #This is the magic function that makes it work so much quicker. 
        
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            #get unique kz values from kpoints_bz
            kz_values = np.unique(Ham_inst.Kpoints_BZ[:,2])
            kz_index = np.argmin(np.abs(np.array(kz_values) - kz))
            print(f"kz_index:{kz_values[kz_index]} (index {kz_index})")

            Eigenvalues = np.array(Ham_inst.Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,Ham_inst.nbands))

            if len(Ham_inst.Orbital_Colour) <1:
                print(f"Error: No orbital colours specified for Hamiltonian {idx}. Please specify orbital colours using Ham.DefineOrbitals([orbital_label_1,orbital_label_2,...],[colour_1,colour_2,...])")
                return 0
            edges = MarchingCube2D(Eigenvalues,isolevel=omega,kz_slice=kz_index,n_repetition=(N_BrillouinZone*2 -1)) #repetition defines how far beyond the 1st BZ we go. If it's 2 that means we sample 3 BZ (-1,0,+1).
            #count edges
            edge_count = 0 #Number of lines can be different for each FS plot. need to count in advance. 
            for i in range(len(edges)):
                edge_count+=len(edges[i])
            #print(edge_count)
            #exit(0)
            segs = np.zeros((edge_count, 2, 2))
            Colours = ["" for i in range(edge_count)]
            #nlines = len(EigVal)*len(EigenStore)
            #Loop over every pair of joining eigenvalues and make an (x1,y1) and (x2,y2) pair for each. ([segs[i][0][0],segs[i][0][1]]) and ([segs[i][1][0],segs[i][1][1]])
            line_counter =0
            for band_edge_index,band_edge in enumerate(edges):
                for edge in band_edge:
                    #These four get the correct shape. edges are just indices of k points (e.g a 16x16 kpoint grid will have coordinates between [0,0] and [15,15]).
                    #We have to first get that to between [0,0] and [1,1] and then we need to multiply them to get them in the range of [-b1,-b2], [+b1,+b2] where b1 and b2 are the reciprocal lattice vectors.
                    # we thus do kx = p1/nkx * b1x + p2/nky * b1y  and kxy = p1/nkx * b2x + p2/nky * b2y for each ky pair. 
                    #for FermiSurface, we only need kx,ky, but the function returns kx,ky,kz. Hence we save the kz to dummy_ and never use it. 
                    segs[line_counter][0][0],segs[line_counter][0][1],dummy_ = Ham_inst.Convert_Kspace_To_XYZ(edge[0][0]/Ham_inst.nkx,edge[0][1]/Ham_inst.nky,0.0) #Start #(edge[0][0]/Ham_inst.nkx)*Ham_inst.reciprocal_vectors[0][0] + (edge[0][1]/Ham_inst.nky)*Ham_inst.reciprocal_vectors[1][0]
                    segs[line_counter][1][0],segs[line_counter][1][1],dummy_ = Ham_inst.Convert_Kspace_To_XYZ(edge[1][0]/Ham_inst.nkx,edge[1][1]/Ham_inst.nky,0.0) #End 
                   
                    #This bit just shifts the Fermi surface to the center of the plot (i.e not [0,0] to [1,1] but [-0.5,-0.5] to [0.5,0.5])
                    shiftocenter = (0.5 +(N_BrillouinZone-1))
                    shiftocenter_kx,shiftocenter_ky,dummy_ = Ham_inst.Convert_Kspace_To_XYZ(shiftocenter,shiftocenter,0.0)
                    #(shiftocenter*Ham_inst.reciprocal_vectors[0][0] + shiftocenter*Ham_inst.reciprocal_vectors[1][0])
                    segs[line_counter][0][0] -= shiftocenter_kx
                    segs[line_counter][0][1] -= shiftocenter_ky

                    segs[line_counter][1][0] -= shiftocenter_kx
                    segs[line_counter][1][1] -= shiftocenter_ky
                    
                    #this bit gets the orbital character - it's a bit of double counting as we have already calculated the eigenvalues once, but its much easier to code this way. 
                    H = Load_Hamiltonian(Ham_inst.GetHoppings(),segs[line_counter][0][0],segs[line_counter][0][1],0,Ham_inst.nbands,Ham_inst.lattice_vectors)
                    eigval,eigvec = LA.eigh(H)
                    if Ham_inst.MixOrbitalWeights == True: #Mix the weight, or just take the maximum 
                        Colours[line_counter] = np.sum([np.multiply(Ham_inst.GetOrbitalWeights(band_edge_index,eigvec), Ham_inst.Orbital_Colour[i]) for i in range(len(Ham_inst.Orbital_Colour))],axis=1)
                    else:
                        Colours[line_counter] = Ham_inst.Orbital_Colour[np.argmax(Ham_inst.GetOrbitalWeights(band_edge_index,eigvec))]
        
                    line_counter+=1
    
            #This is the magic function that makes it work so much quicker. 
            line_segments = LineCollection(segs, linewidths=linewidth[idx%len(linewidth)],colors=Colours,linestyle='solid')
            line_segments.set_antialiased(setAntiAliased)
            self.ax_FS_orbColour.add_collection(line_segments)

        #if N_BrillouinZone > 1:
        if showBrillouinZone:
            BZ_Edges = np.zeros((len(Ham_inst.BZedges), 2, 2))
            BZ_colours  = ["" for i in range(len(Ham_inst.BZedges))]
            for i in range(len(Ham_inst.BZedges)):
                BZ_Edges[i][0][0] = Ham_inst.BZedges[i][0]
                BZ_Edges[i][0][1] = Ham_inst.BZedges[i][1]
                BZ_Edges[i][1][0] = Ham_inst.BZedges[i][3]
                BZ_Edges[i][1][1] = Ham_inst.BZedges[i][4] 
                BZ_colours = "black"
            #print(Ham_inst.BZedges)
            line_segments = LineCollection(BZ_Edges, linewidths=2,colors=BZ_colours,linestyle='dashed')
            self.ax_FS_orbColour.add_collection(line_segments)
        xlim,ylim,zlim = Ham_inst.Convert_Kspace_To_XYZ(N_BrillouinZone-1.5,N_BrillouinZone-1.5,0)# (N_brillouinZone-1 -1/2) the minus half puts zero in the middle, the -(N_Brillouin-1) zone puts the middle Brillouin zone in the middle. 
            
        self.ax_FS_orbColour.set_xlim([-N_BrillouinZone*xlim,+N_BrillouinZone*xlim]) 
        self.ax_FS_orbColour.set_ylim([-N_BrillouinZone*ylim, N_BrillouinZone*ylim])

        self.ax_FS_orbColour.set_xlabel(r"$k_x$")
        self.ax_FS_orbColour.set_ylabel(r"$k_y$")
        self.ax_FS_orbColour.set_aspect('equal', 'box')

    def FermiSurface3D(self,omega=0.0,N_BrillouinZone_kx=1,N_BrillouinZone_ky=1,N_BrillouinZone_kz=1,alpha=1):
        self.fig_FS_3D = plt.figure()
        self.ax_FS_3D = self.fig_FS_3D.add_subplot(111, projection='3d')
        #We have a marching cubes algorithm which goes through the Eigenvalues and gets the isocontour vertices and faces. 
        #It's a big ugly, but  works. 
        #vert has shape vert[band][unique index] = [unique index, [x,y,z]] 
        #Unfortuantely, to make sure the faces new what index to refer to, we had to save the unique index into the vert. 
        #The idea is that each face is a triangle, and each triangle has 3 indices, which correspond to a specific vertex held in vert[band_index][face_index][1]
        #face is a list of shape face[band][face_number] = [index1,index2,index3]. 
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            Eigenvalues = np.array(Ham_inst.Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,Ham_inst.nbands))
            vert, face = MarchingCube3D(Eigenvalues,isolevel=omega,n_repetition_kx=(N_BrillouinZone_kx*2 -1),n_repetition_ky=(N_BrillouinZone_ky*2 -1),n_repetition_kz=(N_BrillouinZone_kz*2 -1)) #repetition defines how far beyond the 1st BZ we go. If it's 2 that means we sample 3 BZ (-1,0,+1).
            #Need to convert the vertices from index to k-space. This only works when the kgrid is defined between -pi and pi.
            for b in range(Ham_inst.nbands):
                for vi,v in enumerate(vert[b]):
                    v[1][0],v[1][1],v[1][2] = Ham_inst.Convert_Kspace_To_XYZ(v[1][0]/Ham_inst.nkx,v[1][1]/Ham_inst.nky,v[1][2]/Ham_inst.nkz)
                    shiftocenter_k1 = (0.5 +(N_BrillouinZone_kx-1))
                    shiftocenter_k2 = (0.5 +(N_BrillouinZone_ky-1))
                    shiftocenter_k3 = (0.5 +(N_BrillouinZone_kz-1))
                    shiftocenter_kx,shiftocenter_ky,shiftocenter_kz = Ham_inst.Convert_Kspace_To_XYZ(shiftocenter_k1,shiftocenter_k2,shiftocenter_k3)
                    #print(b,v,vert[b][])#vert[b][1][1],vert[b][1][2])
                    vert[b][vi][1][0] = v[1][0] - shiftocenter_kx
                    vert[b][vi][1][1] = v[1][1] - shiftocenter_ky
                    vert[b][vi][1][2] = v[1][2] - shiftocenter_kz
  
            #Take each face, which has 3 indices, these correspond to a specific vertex held in vert[band_index][face_index][1] and save those three Vertices as a Poly3DCollection. 
            for b in range(Ham_inst.nbands):
                for f in face[b]:
                    #Okay this is horrible, but it works. 
                    V = [vert[b][f[0]][1],vert[b][f[1]][1],vert[b][f[2]][1]]
                    poly3d = Poly3DCollection(V, alpha=alpha)
                    self.ax_FS_3D.add_collection3d(poly3d)

        #if N_BrillouinZone > 1:
        #if showBrillouinZone:
        BZ_Edges = np.zeros((len(Ham_inst.BZedges), 3, 3))
        BZ_colours  = ["" for i in range(len(Ham_inst.BZedges))]
        for i in range(len(Ham_inst.BZedges)):
            BZ_Edges[i][0][0] = Ham_inst.BZedges[i][0]
            BZ_Edges[i][0][1] = Ham_inst.BZedges[i][1]
            BZ_Edges[i][0][2] = Ham_inst.BZedges[i][2]
            BZ_Edges[i][1][0] = Ham_inst.BZedges[i][3]
            BZ_Edges[i][1][1] = Ham_inst.BZedges[i][4] 
            BZ_Edges[i][1][2] = Ham_inst.BZedges[i][5]
            BZ_colours = "black"
            self.ax_FS_3D.plot([Ham_inst.BZedges[i][0],Ham_inst.BZedges[i][3]],[Ham_inst.BZedges[i][1],Ham_inst.BZedges[i][4]],[Ham_inst.BZedges[i][2],Ham_inst.BZedges[i][5]],color="black",linestyle='dashed')
            #print(Ham_inst.BZedges)
        #line_segments = Line3DCollection(BZ_Edges, linewidths=1,colors=BZ_colours,linestyle='dashed')
        #self.ax_FS_3D.add_collection(line_segments)
 
        self.ax_FS_3D.set_xlabel("kx")
        self.ax_FS_3D.set_ylabel("ky")
        self.ax_FS_3D.set_zlabel("kz")

        xlim,ylim,zlim = Ham_inst.Convert_Kspace_To_XYZ(0.5*N_BrillouinZone_kx+0.1,0.5*N_BrillouinZone_ky+0.1,0.5*N_BrillouinZone_kz +0.1)
        self.ax_FS_3D.set_axis_off()
        self.ax_FS_3D.set_xlim(-xlim,xlim)  
        self.ax_FS_3D.set_ylim(-ylim,ylim)  
        self.ax_FS_3D.set_zlim(-zlim,zlim)  
        self.ax_FS_3D.set_box_aspect([xlim/xlim,ylim/xlim,zlim/xlim]) #doesnt matter which one you normalise too, just normalise to one of them!


        #plt.show()

    #We have a marching cubes algorithm which goes through the Eigenvalues and gets the isocontour vertices and faces. 
    #It's a big ugly, but vert has shape vert[band][unique index] = [unique index, [x,y,z]] 
    #Unfortuantely, to make sure the faces new what index to refer to, we had to save the unique index into the vert. 
    #The idea is that each face is a triangle, and each triangle has 3 indices, which correspond to a specific vertex held in vert[band_index][face_index][1]
    #face is a list of shape face[band][face_number] = [index1,index2,index3]. 
    def FermiSurface3D_orbital_resolved(self,omega=0.0,N_BrillouinZone_kx=1,N_BrillouinZone_ky=1,N_BrillouinZone_kz=1,alpha=1.0):
        
        self.fig_FS_3D_orb = plt.figure()
        self.ax_FS_3D_orb = self.fig_FS_3D_orb.add_subplot(111, projection='3d')
        #This is the magic function that makes it work so much quicker. 
        
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            Eigenvalues = np.array(Ham_inst.Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,Ham_inst.nbands))

            vert, face = MarchingCube3D(Eigenvalues,isolevel=omega,n_repetition_kx=(N_BrillouinZone_kx*2 -1),n_repetition_ky=(N_BrillouinZone_ky*2 -1),n_repetition_kz=(N_BrillouinZone_kz*2 -1))
            for b in range(Ham_inst.nbands):
                for vi,v in enumerate(vert[b]):
                    v[1][0],v[1][1],v[1][2] = Ham_inst.Convert_Kspace_To_XYZ(v[1][0]/Ham_inst.nkx,v[1][1]/Ham_inst.nky,v[1][2]/Ham_inst.nkz)
                    shiftocenter_k1 = (0.5 +(N_BrillouinZone_kx-1))
                    shiftocenter_k2 = (0.5 +(N_BrillouinZone_ky-1))
                    shiftocenter_k3 = (0.5 +(N_BrillouinZone_kz-1))
                    shiftocenter_kx,shiftocenter_ky,shiftocenter_kz = Ham_inst.Convert_Kspace_To_XYZ(shiftocenter_k1,shiftocenter_k2,shiftocenter_k3)
                    #print(b,v,vert[b][])#vert[b][1][1],vert[b][1][2])
                    vert[b][vi][1][0] = v[1][0] - shiftocenter_kx
                    vert[b][vi][1][1] = v[1][1] - shiftocenter_ky
                    vert[b][vi][1][2] = v[1][2] - shiftocenter_kz
            for b in range(Ham_inst.nbands):
                for f in face[b]:
                    #Okay this is horrible, but it works. 
                    V = [vert[b][f[0]][1],vert[b][f[1]][1],vert[b][f[2]][1]]
                    #Find the center
                    cx = (V[0][0] + V[1][0] + V[2][0])/3.0
                    cy = (V[0][1] + V[1][1] + V[2][1])/3.0
                    cz = (V[0][2] + V[1][2] + V[2][2])/3.0

                    H = Load_Hamiltonian(Ham_inst.GetHoppings(),cx,cy,cz,Ham_inst.nbands,Ham_inst.lattice_vectors)
                    eigval,eigvec = LA.eigh(H)
                    if Ham_inst.MixOrbitalWeights == True: #Mix the weight, or just take the maximum 
                        Colour =  np.sum([np.multiply(Ham_inst.GetOrbitalWeights(b,eigvec), Ham_inst.Orbital_Colour[i]) for i in range(len(Ham_inst.Orbital_Colour))],axis=1)
                    else:
                        Colour = Ham_inst.Orbital_Colour[np.argmax(Ham_inst.GetOrbitalWeights(b,eigvec))]
        
                    poly3d = Poly3DCollection(V, alpha=1.0,color=Colour,linewidths=0)
                    self.ax_FS_3D_orb.add_collection3d(poly3d)

         #if N_BrillouinZone > 1:
        #if showBrillouinZone:
        BZ_Edges = np.zeros((len(Ham_inst.BZedges), 3, 3))
        BZ_colours  = ["" for i in range(len(Ham_inst.BZedges))]
        for i in range(len(Ham_inst.BZedges)):
            BZ_Edges[i][0][0] = Ham_inst.BZedges[i][0]
            BZ_Edges[i][0][1] = Ham_inst.BZedges[i][1]
            BZ_Edges[i][0][2] = Ham_inst.BZedges[i][2]
            BZ_Edges[i][1][0] = Ham_inst.BZedges[i][3]
            BZ_Edges[i][1][1] = Ham_inst.BZedges[i][4] 
            BZ_Edges[i][1][2] = Ham_inst.BZedges[i][5]
            BZ_colours = "black"
            self.ax_FS_3D_orb.plot([Ham_inst.BZedges[i][0],Ham_inst.BZedges[i][3]],[Ham_inst.BZedges[i][1],Ham_inst.BZedges[i][4]],[Ham_inst.BZedges[i][2],Ham_inst.BZedges[i][5]],color="black",linestyle='dashed')
            #print(Ham_inst.BZedges)
 
        self.ax_FS_3D_orb.set_xlabel("kx")
        self.ax_FS_3D_orb.set_ylabel("ky")
        self.ax_FS_3D_orb.set_zlabel("kz")

        xlim,ylim,zlim = Ham_inst.Convert_Kspace_To_XYZ(0.5*N_BrillouinZone_kx+0.1,0.5*N_BrillouinZone_ky+0.1,0.5*N_BrillouinZone_kz +0.1)
        self.ax_FS_3D_orb.set_axis_off()
        
        self.ax_FS_3D_orb.set_xlim(-xlim,xlim)  
        self.ax_FS_3D_orb.set_ylim(-ylim,ylim)  
        self.ax_FS_3D_orb.set_zlim(-zlim,zlim)  
        self.ax_FS_3D_orb.set_box_aspect([xlim/xlim,ylim/xlim,zlim/xlim]) #doesnt matter which one you normalise too, just normalise to one of them!
    
    
    def plotGapAtEF(self,N_BrillouinZone=1,kz=0):
        self.fig_SCgap = plt.figure()
        self.ax_SCgap = self.fig_SCgap.add_subplot(111)
        x = []
        y = []
        x2 = []
        y2 = []
        gap = []
        average_gap = 0
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            if Ham_inst.Superconducting != True:
                print(f"Hamiltonian {idx} is not superconducting. No gap to plot.")
                return 0

            Eigenvalues,eigvec = Load_Hamiltonian_grid_NormalState(Ham_inst.GetHoppings(),Ham_inst.nbands,Ham_inst.Kpoints_BZ,Ham_inst.lattice_vectors)
            Eigenvalues = np.array(Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,int(Ham_inst.nbands/2))) #Normal state
            #Eigenvalues = np.array(Ham_inst.Eigenvalues).reshape((Ham_inst.nkx, Ham_inst.nky, Ham_inst.nkz,Ham_inst.nbands))        #convert into a nice cubic x,y,z grid of eigenvalues. 
            kz_values = np.unique(Ham_inst.Kpoints_BZ[:,2])
            kz_index = np.argmin(np.abs(np.array(kz_values) - kz))
            print(f"kz_index:{kz_values[kz_index]} (index {kz_index})")
            edges = MarchingCube2D(Eigenvalues,isolevel=0.0,kz_slice=kz_index,n_repetition=(N_BrillouinZone*2 -1)) #repetition defines how far beyond the 1st BZ we go. If it's 2 that means we sample 3 BZ (-1,0,+1).
            
            edge_count = 0 #Number of lines can be different for each FS plot. need to count in advance. 
            for i in range(len(edges)):
                edge_count+=len(edges[i])
            #print(edge_count)
            #exit(0)
            segs = np.zeros((edge_count, 2, 2))
            Colours = ["" for i in range(edge_count)]
            #nlines = len(EigVal)*len(EigenStore)
            #Loop over every pair of joining eigenvalues and make an (x1,y1) and (x2,y2) pair for each. ([segs[i][0][0],segs[i][0][1]]) and ([segs[i][1][0],segs[i][1][1]])
            line_counter =0
            for band_edge_index,band_edge in enumerate(edges): #band_edge_index is the current band, band edge is the list of lines for that band, some bands are 0 if they don't cross EF.
                for edge in band_edge:
                    #These four get the correct shape. edges are just indices of k points (e.g a 16x16 kpoint grid will have coordinates between [0,0] and [15,15]).
                    #We have to first get that to between [0,0] and [1,1] and then we need to multiply them to get them in the range of [-b1,-b2], [+b1,+b2] where b1 and b2 are the reciprocal lattice vectors.
                    # we thus do kx = p1/nkx * b1x + p2/nky * b1y  and kxy = p1/nkx * b2x + p2/nky * b2y for each ky pair. 
                    #for FermiSurface, we only need kx,ky, but the function returns kx,ky,kz. Hence we save the kz to dummy_ and never use it. 
                    segs[line_counter][0][0],segs[line_counter][0][1],dummy_ = Ham_inst.Convert_Kspace_To_XYZ(edge[0][0]/(Ham_inst.nkx-1),edge[0][1]/(Ham_inst.nky-1),0.0) #Start #(edge[0][0]/Ham_inst.nkx)*Ham_inst.reciprocal_vectors[0][0] + (edge[0][1]/Ham_inst.nky)*Ham_inst.reciprocal_vectors[1][0]
                    segs[line_counter][1][0],segs[line_counter][1][1],dummy_ = Ham_inst.Convert_Kspace_To_XYZ(edge[1][0]/(Ham_inst.nkx-1),edge[1][1]/(Ham_inst.nky-1),0.0) #End 
                   
                    #This bit just shifts the Fermi surface to the center of the plot (i.e not [0,0] to [1,1] but [-0.5,-0.5] to [0.5,0.5])
                    shiftocenter = (0.5 +(N_BrillouinZone-1))
                    shiftocenter_kx,shiftocenter_ky,dummy_ = Ham_inst.Convert_Kspace_To_XYZ(shiftocenter,shiftocenter,0.0)

                    segs[line_counter][0][0] -= shiftocenter_kx
                    segs[line_counter][0][1] -= shiftocenter_ky

                    segs[line_counter][1][0] -= shiftocenter_kx
                    segs[line_counter][1][1] -= shiftocenter_ky

                    Delta = GetSCGap(Ham_inst.GetHoppings(),Ham_inst.nbands,Ham_inst.lattice_vectors,segs[line_counter][0][0],segs[line_counter][0][1],0,Ham_inst.SpinPolarised)
                    x.append(segs[line_counter][0][0])
                    y.append(segs[line_counter][0][1])

                    gap.append(Delta[band_edge_index])
                    average_gap+= gap[-1]
                    line_counter+=1
                        
            c = self.ax_SCgap.scatter(x,y,c=gap,cmap="bwr",lw=0,vmin=-max(gap), vmax=max(gap))
            plt.colorbar(c)
            self.ax_SCgap.set_aspect('equal', 'box')
            print("average gap:",average_gap/len(gap))
            print("min_gap:",np.min(gap))
            print("max_gap:",np.max(gap))
        return average_gap/len(gap)

    def PlotSpecificHeat(self,T_start,T_end,T_step,DivideByT=False):
        self.fig_SpecHeat = plt.figure()
        self.ax_SpecHeat = self.fig_SpecHeat.add_subplot(111)
        Temp = np.arange(T_start,T_end,T_step)
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            if Ham_inst.Superconducting == True:
                Cv_NSC = []
                Cv_SC = []
                for T in Temp:
                    Cv_nsc = Ham_inst.CalcSpecificHeat(T,Superconducting=False)
                    if DivideByT == True: 
                        Cv_nsc = Cv_nsc/T
                    Cv_NSC.append(Cv_nsc)
                    Cv_sc = Ham_inst.CalcSpecificHeat(T,Superconducting=True)
                    if DivideByT == True: 
                        Cv_sc = Cv_sc/T
                    Cv_SC.append(Cv_sc)
                    print(T,Cv_sc,Cv_nsc)

                self.ax_SpecHeat.plot(Temp,Cv_NSC,ls="--",c="black",label="Normal State")
                self.ax_SpecHeat.plot(Temp,Cv_SC,c="black",label="Superconducting State")
                Clist = [Cv_NSC,Cv_SC]
            else:
                Cv = []
                for T in Temp:
                    C = Ham_inst.CalcSpecificHeat(T)
                    if DivideByT == True: 
                        C = C/T
                    Cv.append(C)
                    print(T,C)

                self.ax_SpecHeat.plot(Temp,Cv,ls="-",c="black",label="Normal State")
                Clist = [Cv]
            
            self.ax_SpecHeat.legend()
            self.ax_SpecHeat.set_xlabel("T (K)")
            self.ax_SpecHeat.set_ylabel("Cv/T (eV/K)")
            return Temp,Clist



    def SpectralFunction(self,omega,Gamma,orbitals = [1],kz=0.0):
        for idx, Ham_inst in enumerate(self.tbHamiltonian):
            if len(orbitals) != Ham_inst.nbands:
                if (((len(orbitals) == 1) and (orbitals[0] == 1)) == False): #dont send error message for default case.
                    print(f"orbitals array must be a list of length nbands ({Ham_inst.nbands})")
                    print(f"your orbital:{orbitals}")
                    print(f"setting all orbitals to 1 ")
                    orbitals = np.ones(Ham_inst.nbands)
                    print(f"new orbital:{orbitals}")
                else:
                    orbitals = np.ones(Ham_inst.nbands)
            else:
                orbitals = np.array(orbitals)
                print(f"orbitals:{orbitals}")

            self.fig_SF,self.ax_SF = plt.subplots()
            self.ax_SF.set_aspect('equal', 'box')
            x = np.linspace(-np.pi, np.pi, self.nkx)
            y = np.linspace(-np.pi, np.pi, self.nky)
            
            AK = np.zeros((Ham_inst.nkx,Ham_inst.nky),dtype=np.float64)

            kz_index = 0#int(((kz%(2*np.pi) - np.pi) / (2 * np.pi)) * self.nkz)
            #print(f"closest kz is:{kz_index*2*np.pi/self.nkz + np.pi}")            
            AK = CalcSpectralFunction(omega,Gamma,Ham_inst.Eigenvalues,Ham_inst.EigenVectors,kz=0)
            self.ax_SF.matshow(AK)
                
#@timeit
@nb.njit
def CalcDOS(omega,Gamma,Energy_store,Evectors_store,superconducting=False):
    nk,nbands = np.shape(Energy_store)
    if superconducting == True:
        superconducting_factor = 2
    else:
        superconducting_factor = 1
    DOS = 0
    for k in range(len(Energy_store)):
        for band in range(nbands):
            denominator = 1.0/(omega +1j*Gamma - Energy_store[k][band])
            for s in range(int(nbands)):
                DOS += Evectors_store[k][s][band]*np.conj(Evectors_store[k][s][band])*denominator
    DOS = -(1.0/np.pi)*np.imag(DOS)/(nk)
    return DOS


#@timeit
@nb.njit
def CalcDOS_Orbital_resolved(omega,Gamma,Energy_store,Evectors_store,superconducting=False):
    nk,nbands = np.shape(Energy_store)
    if superconducting == True:
        superconducting_factor = 2
    else:
        superconducting_factor = 1
    DOS = 0
    orbital_DOS = np.zeros(int(nbands/superconducting_factor)+1,dtype=np.float64)
    for k in range(len(Energy_store)):
        for band in range(nbands):
            denominator = 1.0/(omega +1j*Gamma - Energy_store[k][band])
            for s in range(int(nbands/superconducting_factor)):
                dos = Evectors_store[k][s][band]*np.conj(Evectors_store[k][s][band])*denominator
                orbital_DOS[s] += np.imag(dos)
                orbital_DOS[-1] += np.imag(dos)
    for i in range(len(orbital_DOS)):
        orbital_DOS[i] = -(1.0/np.pi)*orbital_DOS[i]/(nk)
    return orbital_DOS



@nb.njit
def CalcSpectralFunction(omega,Gamma,Energy_store,Evectors_store,kz=0):
    nkx,nky,nkz,nbands = np.shape(Energy_store)
    
    AK = np.zeros((nkx,nky),dtype=np.float64)
    for kx in range(nkx):
        for ky in range(nky):
            DOS = 0
            for band in range(nbands):
                denominator = 1.0/(omega +1j*Gamma - Energy_store[kx][ky][kz][band])
                for s in range(nbands):
                    DOS += Evectors_store[kx][ky][kz][s][band]*np.conj(Evectors_store[kx][ky][kz][s][band])*denominator
            AK[kx][ky] = -(1.0/np.pi)*np.imag(DOS)/(nkx*nky*nkz)
    return AK



@nb.njit
def adapt(x1,x2,isolevel):
    """
    takes two values and linearly interpolates between them to find the point where the isolevel intersects the line between them.
    """
    return (isolevel-x1)/(x2-x1)

#@nb.njit
def MarchingCube2D(Energy_store,isolevel,kz_slice=0,n_repetition=1):
    cubesize = 1
    nkx,nky,nkz,nbands = np.shape(Energy_store)
    edges = [[] for i in range(nbands)]
    for b in range(nbands):
        vmin = np.min(Energy_store[:,:,:,b])
        vmax = np.max(Energy_store[:,:,:,b])
        if vmin < isolevel < vmax: #dont loop over empty contours
            for x in range(n_repetition*nkx):
                for y in range(n_repetition*nky):
                    x0y0 = Energy_store[x%nkx][y%nky][kz_slice][b]
                    x0y1 = Energy_store[x%nkx][(y+1)%nky][kz_slice][b]
                    x1y0 = Energy_store[(x+1)%nkx][(y)%nky][kz_slice][b]
                    x1y1 = Energy_store[(x+1)%nkx][(y+1)%nky][kz_slice][b]
                    case_val = ((x0y0 > isolevel) << 0 |
                                (x0y1 > isolevel) << 1 |
                                (x1y0 > isolevel) << 2 |
                                (x1y1 > isolevel) << 3)
                    if case_val == 0:
                        continue
                    elif case_val == 15:
                        continue
                    elif case_val == 1:
                        #single corner
                        v1 = [x + cubesize*adapt(x0y0,x1y0,isolevel),y]
                        v2 = [x,y + cubesize*adapt(x0y0,x0y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) #swaps v1 and v2 if case is 14. do this for readability of code.
                        continue
                    elif case_val == 2:
                        #single corner
                        v1 = [x, y + cubesize*adapt(x0y0,x0y1,isolevel)]
                        v2 = [x+ cubesize*adapt(x0y1,x1y1,isolevel),y+cubesize]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 3:
                        #Vertical split
                        v1 = [x+cubesize*adapt(x0y0, x1y0,isolevel), y]
                        v2 = [x+cubesize*adapt(x0y1, x1y1,isolevel), y+cubesize]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 4:
                        #single corner
                        v1 = [x+cubesize, y + cubesize*adapt(x1y0,x1y1,isolevel)]
                        v2 = [x+cubesize*adapt(x0y0,x1y0,isolevel),y]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 5:
                        #Horizontal split
                        v1 = [x, y+cubesize*adapt(x0y0, x0y1,isolevel)]
                        v2 = [x+cubesize, y+cubesize*adapt(x1y0, x1y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 6:
                    # Two opposite corners, copy cases 2 and 4
                        v1 = [x+cubesize, y+cubesize*adapt(x1y0, x1y1,isolevel)]
                        v2 = [x+cubesize*adapt(x0y0, x1y0,isolevel), y]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        v1 = [x, y+cubesize*adapt(x0y0, x0y1,isolevel)]
                        v2 = [x+cubesize*adapt(x0y1, x1y1,isolevel), y+cubesize]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 7:
                    #single corner, reverse of 8
                        v2 = [x+cubesize*adapt(x0y1, x1y1,isolevel), y+cubesize]
                        v1 = [x+cubesize, y+cubesize*adapt(x1y0, x1y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 8:
                    #single corner
                        v1 = [x+cubesize*adapt(x0y1, x1y1,isolevel), y+cubesize]
                        v2 = [x+cubesize, y+cubesize*adapt(x1y0, x1y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 9:
                    # Two opposite corners, copy cases 1 and 8
                        v1 = [x + cubesize*adapt(x0y0, x1y0,isolevel), y]
                        v2 = [x, y + cubesize*adapt(x0y0, x0y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        v1 = [x + cubesize*adapt(x0y1, x1y1,isolevel), y + cubesize]
                        v2 = [x + cubesize, y + cubesize*adapt(x1y0, x1y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 10:
                    #Horizontal split, reverse of 5
                        v2 = [x, y+cubesize*adapt(x0y0, x0y1,isolevel)]
                        v1 = [x+cubesize, y+cubesize*adapt(x1y0, x1y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 11:
                    #single corner, reverse of 4
                        v2 = [x+cubesize, y + cubesize*adapt(x1y0,x1y1,isolevel)]
                        v1 = [x+cubesize*adapt(x0y0,x1y0,isolevel),y]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 12:
                    #Vertical split, reverse of 3
                        v2 = [x+cubesize*adapt(x0y0, x1y0,isolevel), y]
                        v1 = [x+cubesize*adapt(x0y1, x1y1,isolevel), y+cubesize]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 13:
                    #single corner, reverse of 2
                        v2 = [x, y + cubesize*adapt(x0y0,x0y1,isolevel)]
                        v1 = [x+cubesize*adapt(x0y1,x1y1,isolevel),y+cubesize]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue
                    elif case_val == 14:
                    #single corner, reverse of 1
                        v2 = [x + cubesize*adapt(x0y0,x1y0,isolevel),y]
                        v1 = [x,y + cubesize*adapt(x0y0,x0y1,isolevel)]
                        edges[b].append([np.real(v1),np.real(v2)]) 
                        continue

    return edges

@nb.njit
def interpolation_value(v1,v2,t):
    return np.real((v1-t)/(v1-v2))
 
 #Calculate x,y and z coordinates using linear interpolation
@nb.njit
def linear_interpolation(edge,data,top,left,depth,isolevel):
 tval = 0
 point = None
  #edge 0
 if (edge == 0):
    if (((left,top,depth),(left+1,top,depth)) in cache):
        point = cache[((left,top,depth),(left+1,top,depth))]
    else:
        tval = interpolation_value (data[left%len(data),(top)%len(data[0]),(depth)%len(data[0][0])],data[(left+1)%len(data),top%len(data[0]),depth%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+tval,top,depth)
        cache[((left,top,depth),(left+1,top,depth))] = point
    return point

#edge 1
 if (edge == 1):
    if (((left+1,top,depth),(left+1,top+1,depth)) in cache):
        point = cache[((left+1,top,depth),(left+1,top+1,depth))]
    else:
        tval = interpolation_value (data[(left+1)%len(data),top%len(data[0]),depth%len(data[0][0])],data[(left+1)%len(data),(top+1)%len(data[0]),depth%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+1,top+tval,depth)
        cache[((left+1,top,depth),(left+1,top+1,depth))] = point
    return point

#edge 2
 if (edge == 2):
    if (((left,top+1,depth),(left+1,top+1,depth)) in cache):
        point = cache[((left,top+1,depth),(left+1,top+1,depth))]
    else:
        tval = interpolation_value (data[left%len(data),(top+1)%len(data[0]),depth%len(data[0][0])],data[(left+1)%len(data),(top+1)%len(data[0]),depth%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+tval,top+1,depth)
        cache[((left,top+1,depth),(left+1,top+1,depth))] = point
    return point

#edge 3
 if (edge == 3):
    if (((left,top,depth),(left,top+1,depth)) in cache):
        point = cache[((left,top,depth),(left,top+1,depth))]
    else:
        tval = interpolation_value (data[left%len(data),top%len(data[0]),depth%len(data[0][0])],data[left%len(data[0][0]),(top+1)%len(data[0]),depth%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left,top+tval,depth)
        cache[((left,top,depth),(left,top+1,depth))] = point
    return point

#edge 4
 if (edge == 4):
    if (((left,top,depth+1),(left+1,top,depth+1)) in cache):
        point = cache[((left,top,depth+1),(left+1,top,depth+1))]
    else:
        tval = interpolation_value (data[left%len(data),top%len(data[0]),(depth+1)%len(data[0][0])],data[(left+1)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+tval,top,depth+1)
        cache[((left,top,depth+1),(left+1,top,depth+1))] = point
    return point

#edge 5
 if (edge == 5):
    if (((left+1,top,depth+1),(left+1,top+1,depth+1)) in cache):
        point = cache[((left+1,top,depth+1),(left+1,top+1,depth+1))]
    else:
        tval = interpolation_value (data[(left+1)%len(data),top%len(data[0]),(depth+1)%len(data[0][0])],data[(left+1)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+1,top+tval,depth+1)
        cache[((left+1,top,depth+1),(left+1,top+1,depth+1))] = point
    return point

#edge 6
 if (edge == 6):
    if (((left,top+1,depth+1),(left+1,top+1,depth+1)) in cache):
        point = cache[((left,top+1,depth+1),(left+1,top+1,depth+1))]
    else:
        tval = interpolation_value (data[(left)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],data[(left+1)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+tval,top+1,depth+1)
        cache[((left,top+1,depth+1),(left+1,top+1,depth+1))] = point
    return point

#edge 7
 if (edge == 7):
    if (((left,top,depth+1),(left,top+1,depth+1)) in cache):
        point = cache[((left,top,depth+1),(left,top+1,depth+1))]
    else:
        tval = interpolation_value (data[(left)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])],data[(left)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left,top+tval,depth+1)
        cache[((left,top,depth+1),(left,top+1,depth+1))] = point
    return point

#edge 8
 if (edge == 8):
    if (((left,top,depth),(left,top,depth+1)) in cache):
        point = cache[((left,top,depth),(left,top,depth+1))]
    else:
        tval = interpolation_value (data[(left)%len(data),(top)%len(data[0]),(depth)%len(data[0][0])],data[(left)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left,top,depth+tval)
        cache[((left,top,depth),(left,top,depth+1))] = point
    return point

#edge 9
 if (edge == 9):
    if (((left+1,top,depth),(left+1,top,depth+1)) in cache):
        point = cache[((left+1,top,depth),(left+1,top,depth+1))]
    else:
        tval = interpolation_value (data[(left+1)%len(data),(top)%len(data[0]),(depth)%len(data[0][0])],data[(left+1)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+1,top,depth+tval)
        cache[((left+1,top,depth),(left+1,top,depth+1))] = point
    return point

#edge 10
 if (edge == 10):
    if (((left+1,top+1,depth),(left+1,top+1,depth+1)) in cache):
        point = cache[((left+1,top+1,depth),(left+1,top+1,depth+1))]
    else:
        tval = interpolation_value (data[(left+1)%len(data),(top+1)%len(data[0]),(depth)%len(data[0][0])],data[(left+1)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left+1,top+1,depth+tval)
        cache[((left+1,top+1,depth),(left+1,top+1,depth+1))] = point
    return point

#edge 11
 if (edge == 11):
    if (((left,top+1,depth),(left,top+1,depth+1)) in cache):
        point = cache[((left,top+1,depth),(left,top+1,depth+1))]
    else:
        tval = interpolation_value (data[(left)%len(data),(top+1)%len(data[0]),(depth)%len(data[0][0])],data[(left)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])],isolevel)
        if (tval is None):
            return None
        point = (left,top+1,depth+tval)
        cache[((left,top+1,depth),(left,top+1,depth+1))] = point
    return point
 
#Reference Marching Cubes: Cases Slide (Lecture Notes)
#Build the 8-bit conversion code (Modified it to generate the decimal value)
def getContourCase(top,left,depth, isolevel,data):
   x = 0  
   if (isolevel < data[left%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])]):
        x = 128
   if (isolevel < data[(left+1)%len(data),(top+1)%len(data[0]),(depth+1)%len(data[0][0])]):
        x = x + 64
   if (isolevel < data[(left+1)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])]):
        x = x + 32
   if (isolevel < data[(left)%len(data),(top)%len(data[0]),(depth+1)%len(data[0][0])]):
        x = x + 16
   if (isolevel < data[(left)%len(data),(top+1)%len(data[0]),(depth)%len(data[0][0])]):
        x = x + 8
   if (isolevel < data[(left+1)%len(data),(top+1)%len(data[0]),(depth)%len(data[0][0])]):
        x = x + 4
   if (isolevel < data[(left+1)%len(data),(top)%len(data[0]),(depth)%len(data[0][0])]):
        x = x + 2
   if (isolevel < data[(left)%len(data),(top)%len(data[0]),(depth)%len(data[0][0])]):
        x = x + 1
   case_value = triTable[x] 
   return case_value  
 
#Build Marching Cubes Lookup table
#This table is straight from Paul Bourke's site (Source: http://paulbourke.net/geometry/polygonise/)

triTable =[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
            [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
            [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
            [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
            [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
            [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
            [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
            [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
            [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
            [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
            [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
            [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
            [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
            [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
            [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
            [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
            [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
            [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
            [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
            [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
            [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
            [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
            [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
            [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
            [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
            [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
            [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
            [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
            [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
            [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
            [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
            [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
            [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
            [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
            [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
            [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
            [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
            [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
            [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
            [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
            [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
            [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
            [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
            [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
            [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
            [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
            [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
            [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
            [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
            [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
            [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
            [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
            [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
            [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
            [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
            [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
            [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
            [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
            [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
            [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
            [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
            [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
            [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
            [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
            [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
            [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
            [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
            [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
            [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
            [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
            [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
            [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
            [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
            [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
            [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
            [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
            [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
            [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
            [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
            [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
            [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
            [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
            [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
            [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
            [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
            [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
            [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
            [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
            [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
            [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
            [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
            [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
            [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
            [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
            [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
            [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
            [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
cache={}
#See https://github.com/alvin-yang68/Marching-Cubes/blob/main/Marching_Cubes.ipynb
def MarchingCube3D(Energy_store,isolevel,n_repetition_kx=1,n_repetition_ky=1,n_repetition_kz=1):
    nkx,nky,nkz,nbands = Energy_store.shape
    Energy_store =  np.transpose(Energy_store, (3, 0, 1, 2)) #get the data in bands, kx,ky,kz

    vertex_array = [[] for i in range(nbands)]
    t1 = time.time()
    Faces = [[] for i in range(nbands)]
    for b in range(nbands):
        vmin = np.min(Energy_store[b,:,:,:])
        vmax = np.max(Energy_store[b,:,:,:])
        if vmin < isolevel < vmax: #dont loop over empty contours
            vertex_counter = 0
            for left in range(0, n_repetition_kx*nkx):
                for top in range(0, n_repetition_ky*nky):
                    for depth in range(0, n_repetition_kz*nkz):
                        case_val = getContourCase(top,left,depth,isolevel,Energy_store[b])
                        k = 0
                        while (case_val [k] != -1):
                            v1 =  linear_interpolation(case_val [k],Energy_store[b],top,left,depth,isolevel)
                            v1 = np.array(v1)
                            if v1 is None:
                                k = k + 3
                                continue
                            v2  = linear_interpolation(case_val [k+1],Energy_store[b],top,left,depth,isolevel)
                            v2 = np.array(v2)
                            if v2 is None:
                                k = k + 3
                                continue
                            v3 =  linear_interpolation(case_val [k+2],Energy_store[b],top,left,depth,isolevel)
                            v3 = np.array(v3)
                            if v3 is None:
                                k = k + 3
                                continue

                            k = k + 3
                            tmp = [0, 0, 0]
                            vertex_array_store = [vertex_array[b][i][1] for i in range(len(vertex_array[b]))]
                            
                            v1_index = next((i for i, arr in enumerate(vertex_array_store) if np.array_equal(arr, v1)), None)
                            v2_index = next((i for i, arr in enumerate(vertex_array_store) if np.array_equal(arr, v2)), None)
                            v3_index = next((i for i, arr in enumerate(vertex_array_store) if np.array_equal(arr, v3)), None)
                            if v1_index is None:
                                vertex_array[b].append(np.array([vertex_counter,v1]))
                                tmp[0] = vertex_counter
                                vertex_counter += 1
                            else:
                                tmp[0] = vertex_array[b][v1_index][0]
                            if v2_index is None:
                                vertex_array[b].append(np.array([vertex_counter,v2]))
                                tmp[1] = vertex_counter
                                vertex_counter += 1
                            else:
                                tmp[1] = vertex_array[b][v2_index][0]
                            if v3_index is None:
                                vertex_array[b].append(np.array([vertex_counter,v3]))
                                tmp[2] = vertex_counter
                                vertex_counter += 1
                            else:
                                tmp[2] = vertex_array[b][v3_index][0]

                            Faces[b].append(tmp)
    t2 = time.time()
    print("\nTime taken by maching cube 3D algorithm\n"+'-'*40+"\n{} s".format(t2-t1))
    #print(vertex_array)
    return np.array(vertex_array), np.array(Faces)


def get_lattice_pointGroup(a_vecs, eps=1E-10):
    """This routine returns only the point group of the lattite rather
      than the space group of the given crystal structure.
    Args:
        a_vecs (array-like): The 2D array that contains the parent lattice
            vectors as row vectors.
        eps (float, optional): Finite precision tolerance
    Returns:
        lattpg_op (array-like): The point group for the lattice in
        cartesian coordinates.
    """

    inverse_avecs = LA.inv(np.array(a_vecs))
    #print(inverse_avecs)

    # Store the norms of the three lattice vectors
    norm_avecs = []
    for i in range(3):
        norm_avecs.append(LA.norm(a_vecs[i]).tolist())

    #print("norm_avecs",norm_avecs)
    # Decide how many lattice points to look in each direction to get all the
    # points in a sphere that contains all of the longest _primitive_ vectors
    cell_volume = abs(np.dot(a_vecs[0],np.cross(a_vecs[1],a_vecs[2])))
    max_norm = max([LA.norm(i) for i in a_vecs])

    #print("cell_volume",cell_volume)
    #print("max_norm",max_norm)
    n1 = math.ceil(max_norm*LA.norm(np.cross(a_vecs[1],a_vecs[2])/cell_volume+eps))
    n2 = math.ceil(max_norm*LA.norm(np.cross(a_vecs[2],a_vecs[0])/cell_volume+eps))
    n3 = math.ceil(max_norm*LA.norm(np.cross(a_vecs[0],a_vecs[1])/cell_volume+eps))

    #print("n1n2n3",n1,n2,n3)
    r_vecs = []
    r_lengths = []

    a_vecs = np.array(a_vecs)
    # Store the R vectors that lie within the sphere
    num_rs = 0
    for i in range(-int(round(n1)), int(round(n1))+1):
        for j in range(-int(round(n2)), int(round(n2))+1):
            for k in range(-int(round(n3)), int(round(n3))+1):
                this_vector = i*a_vecs[0] + j*a_vecs[1] + k*a_vecs[2]
                length = LA.norm(this_vector)
                if (length > max_norm + eps):
                    continue # This vector is outside sphere
                num_rs += 1
                r_vecs.append(this_vector.tolist())
                r_lengths.append(length)
    #print("r_Vecs size",len(r_vecs),len(r_lengths))

    counter=0
    for r_vec in r_vecs:
        #print(counter,"r_vecs,r_lengths",r_vec,r_lengths[counter])
        counter+=1

    # Try all R vector triplets in the sphere and see which ones are valid
    # rotations of the original basis vectors.
    #
    # The length of all vectors must be preserved under a unitary
    # transformation so skip any trial vectors that aren't the same
    # length as the original. We also skip any set of vectors that
    # have the right lengths but do not form a parallelpiped that has
    # the same volume as the original set. Also, note that the we skip
    # sets of vectors that contain the same vector more than once
    # (i.e. the indices i, j, k must be unique).

    from itertools import permutations


    num_ops = 0
    lattpg_op = []
    for i,j,k in permutations(range(num_rs),3):
        if (abs(r_lengths[i] - norm_avecs[0]) > eps) or (abs(r_lengths[j] - norm_avecs[1]) > eps) or (abs(r_lengths[k] - norm_avecs[2]) > eps) or (abs(cell_volume - abs(np.linalg.det([r_vecs[i],r_vecs[j],r_vecs[k]]))) > eps):
            continue
        # Form the new set of "rotated" basis vectors
        #print(i,j,k,abs(np.linalg.det([r_vecs[i],r_vecs[j],r_vecs[k]])))
        new_vectors = [r_vecs[i],r_vecs[j],r_vecs[k]]
        # If the transformation matrix that takes the original set to the new set is
        # an orthogonal matrix then this rotation is a point symmetry of the lattice.
        rotation_matrix = np.matmul(inverse_avecs,new_vectors)
        # Check orthogonality of rotation matrix by [R][R]^T = [1]
        test_matrix = np.matmul(rotation_matrix,np.transpose(rotation_matrix))
        if (np.allclose(test_matrix, [[1,0,0],[0,1,0],[0,0,1]], rtol=0,atol=eps)): # Found valid rotation
            #print(num_ops,i,j,k)
            for i in range(3):
                for j in range(3):
                    if (np.abs(rotation_matrix[i][j]) < 1e-6):
                        rotation_matrix[i][j]=0

            num_ops +=  1 # Count number of rotations
            lattpg_op.append(rotation_matrix.tolist())
    return(lattpg_op)

def cornerIsUnchanged(corner,g):
    eps = 1e-6
    V = np.array(corner)
    gV = np.dot(g,V)
    ##create connecting vector
    Vp=gV-V
    if (LA.norm(Vp)<eps):
        return True
    else:
        return False

def MakeIrreducibleBZBoundaries(lattpg_op,corners,edges):
    Gs = np.zeros((len(lattpg_op)))
    Gs.fill(1)

    newCorners = corners
    newEdges = edges
    for corner in corners:
        #print("helloE2",corner)
        for gi,g in enumerate(lattpg_op):
            if (Gs[gi] == 1):
                if (cornerIsUnchanged(corner,g)):
                    Gs[gi] = 1
                else:
                    Gs[gi] = 0

                    newCorners,newEdges =ApplySymmetryOpperation(corner,g,newCorners,newEdges)
    return newCorners,newEdges

def ApplySymmetryOpperation(corner,g,corners,edges):
    eps=1e-6
    #Step 2
    ### Define plane normal to Vp, and delete all points below the plane.
    #V = vector that connects original corner to transformed corner.

    V = np.array(corner)
    gV = np.dot(g,V)
    #print("V",V)
    #print("g",g)
    #print("gV",gV)
    ##create connecting vector
    Vp=gV-V
    if (LA.norm(Vp)<eps):
        return corners,edges



    Vmid = V+(Vp/2)
    ##create plane
    newCorners = []
    for corner in corners:
        #print(Vp,Vmid,corner)
        PlaneEq = PlaneEquation(Vp,Vmid,corner)
        #print("CHECK PlaneEq",corner,PlaneEq)
        #print(PlaneEq)

        if (PlaneEq <= 0):
            newCorners.append(corner)
            #print("after plane corners:",corner)


    newEdges = []
    cornerOnNewPlane = []
    for edge in edges:
        #print(edge[1],edge[4],V_rot[1])
        edge = np.array(edge)
        edge_start = np.array(edge[:3])
        edge_end = np.array(edge[3:])

        #print("EDGES", edge_start, edge_end)

        PlaneEq_start = PlaneEquation(Vp,Vmid,edge_start)
        PlaneEq_end = PlaneEquation(Vp,Vmid,edge_end)
        #print("PLANE_EQ EDGES",PlaneEq_start,PlaneEq_end)
        if ((PlaneEq_start > 0) and (PlaneEq_end> 0)):
            a=0
            #print("less than = remove",edge_start,edge_end)
        elif((PlaneEq_start > 0) and (np.abs(PlaneEq_end)<eps)):
            #print("edge touching corner, remove")
            cornerOnNewPlane.append(edge_end)
        elif((np.abs(PlaneEq_start)<eps) and (PlaneEq_end> 0)):
            #print("edge touching corner, remove")
            cornerOnNewPlane.append(edge_start)
        elif((PlaneEq_start > 0) and (PlaneEq_end< 0)):
            #print("edge_cut!",np.dot(Vp,Vmid),np.dot(Vp,edge_end-edge_start))
            alpha = PlaneEquation_intercept(Vp,Vmid,edge_start,edge_end)
            #print(alpha)
            #print(edge)
            edgefix = [edge[0] + alpha*(edge[3]-edge[0]),
                       edge[1] + alpha*(edge[4]-edge[1]),
                       edge[2] + alpha*(edge[5]-edge[2]),
                       edge[3],
                       edge[4],
                       edge[5]]

            newEdges.append(edgefix)
            #print("EDGEFIX_1",newEdges[-1])
            newvertex  = [edge[0] + alpha*(edge[3]-edge[0]),
                          edge[1] + alpha*(edge[4]-edge[1]),
                          edge[2] + alpha*(edge[5]-edge[2])]
            newCorners.append(newvertex)
            #print("corner from 1",newvertex)

            cornerOnNewPlane.append(newvertex)
        elif((PlaneEq_start<0) and (PlaneEq_end > 0)):
            #print("edge_cut!")
            alpha = PlaneEquation_intercept(Vp,Vmid,edge_end,edge_start)
            #print(alpha)
            #print(edge)
            edgefix = [edge[0],
                       edge[1],
                       edge[2],
                       edge[3] - alpha*(edge[3]-edge[0]),
                       edge[4] - alpha*(edge[4]-edge[1]),
                       edge[5] - alpha*(edge[5]-edge[2])]
            newEdges.append(edgefix)
            #print("EDGEFIX_2",newEdges[-1])
            newvertex = [edge[3] - alpha*(edge[3]-edge[0]),
                                  edge[4] - alpha*(edge[4]-edge[1]),
                                  edge[5] - alpha*(edge[5]-edge[2])]
            newCorners.append(newvertex)
            cornerOnNewPlane.append(newvertex)
            #print(cornerOnNewPlane[-1])
        else:
            newEdges.append(edge)
            #print("EDGEFIX_3",newEdges[-1])


    #To join the new corners,
    #1) find the center of the points on the plane,
    #2) create vectors connecting the center to the points
    #3) calculate the angle between one of the vectors, and all other vectors.
    #4) order those vectors with respect to the angle between the vectors.
    #5) join the dots to create the new edges.

    ##calculate center of points on the plane so we can join the vertices.
    #print("cornerOnNewPlane",cornerOnNewPlane)

    if (len(cornerOnNewPlane) !=0):
        center = np.array([0.0,0.0,0.0])
        for point in cornerOnNewPlane:
            #print(point)
            center +=np.array(point)

        #print("center before",center)
        #print("len corner on new plane",len(cornerOnNewPlane),cornerOnNewPlane)
        center = np.array(center)/len(cornerOnNewPlane)
        #print("center after",center)

        ##create vectors connecting center to points
        startingVectorV = cornerOnNewPlane[0] - center
        #print("startingVectorV",startingVectorV)
        anglelist  = []
        ##find angle between that vector, and all other similar vectors on plane
        for point in cornerOnNewPlane:
            VectorV = point-center
            dot = np.dot(VectorV,startingVectorV)
            cross = np.cross(VectorV,startingVectorV)

            normV = V/LA.norm(V)
            normcross = np.dot(normV,cross)
            angle = np.arctan2(dot,normcross)*180/np.pi -90
            if angle<0:
                angle+=360
            anglelist.append(angle)
            #print(VectorV,startingVectorV,angle)

        angleindex = np.argsort(anglelist)
        cornerOnNewPlane = np.array(cornerOnNewPlane)[angleindex[:]]
        #for C in cornerOnNewPlane:
        #    print("new plane",C)
        for i in range(len(cornerOnNewPlane)-1):
            newEdges.append([cornerOnNewPlane[i][0],
                           cornerOnNewPlane[i][1],
                           cornerOnNewPlane[i][2],
                           cornerOnNewPlane[i+1][0],
                           cornerOnNewPlane[i+1][1],
                           cornerOnNewPlane[i+1][2]])

        #close the loop
        newEdges.append([cornerOnNewPlane[-1][0],
                           cornerOnNewPlane[-1][1],
                           cornerOnNewPlane[-1][2],
                           cornerOnNewPlane[0][0],
                           cornerOnNewPlane[0][1],
                           cornerOnNewPlane[0][2]])


    return newCorners,newEdges

## The equation for a plane.
## N is the vector which has a normal plane passing through it. P is the point that the plane passes through
## P is the point that the plane passes through (e.g the midpoint of the vector N)
## a is the new point. If it's on the plane this returns 0, if it's below we get a negative value, if above positive.
def PlaneEquation(N,P,a):
    return N[0]*(a[0]-P[0]) + N[1]*(a[1]-P[1]) + N[2]*(a[2]-P[2])

## Given a vector (a) that crosses through the plane, we can use this equation to find out what scale we have to multiply the vector by to hit the plane, and thus figure out the x,y,z points.
def PlaneEquation_intercept(N,P,C0,C1):
    alpha = (N[0]*P[0] + N[1]*P[1] + N[2]*P[2] - N[0]*C0[0] - N[1]*C0[1]-N[2]*C0[2])/(N[0]*(C1[0]-C0[0]) + N[1]*(C1[1]-C0[1]) + N[2]*(C1[2]-C0[2]))
    return alpha

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#SOC matrices
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################


Lx_s = np.array([[0.0]],dtype=np.complex128)
Ly_s = np.array([[0.0]],dtype=np.complex128)
Lz_s = np.array([[0.0j]],dtype=np.complex128)

# px,py,pz
Lx_p = np.array([[0.0, 0.0, 0.0],
                 [0.0, 0.0, -1j],
                 [0.0j, 1j, 0.0]],dtype=np.complex128)

# px,py,pz
Ly_p = np.array([[0.0, 0.0, 1j],
                 [0.0, 0.0, 0.0],
                 [-1j, 0.0, 0.0j]],dtype=np.complex128)

# px,py,pz
Lz_p = np.array([[0.0, -1j, 0.0],
                 [+1j, 0.0, 0.0],
                 [0.0, 0.0, 0.0]],dtype=np.complex128)

# dz2,dxz,dyz,dx2y2,dxy
Lx_d = np.array([[0.0,              0.0, np.sqrt(3.0)*1j, 0.0, 0.0],
                 [0.0,              0.0, 0.0,             0.0, 1j],
                 [-np.sqrt(3.0)*1j, 0.0, 0.0,             -1j, 0.0],
                 [0.0,              0.0,  1j,             0.0, 0.0],
                 [0.0,              -1j, 0.0,             0.0, 0.0]],dtype=np.complex128)

# dz2,dxz,dyz,dx2y2,dxy
Ly_d = np.array([[0.0,             -np.sqrt(3)*1j, 0.0, 0.0, 0.0],
                 [np.sqrt(3.0)*1j, 0.0,            0.0, -1j, 0.0],
                 [0.0,             0.0,            0.0, 0.0, 1j],
                 [0.0,             1j,             0.0, 0.0, 0.0],
                 [0.0,             0.0,            -1j, 0.0, 0.0]],dtype=np.complex128)

# dz2,dxz,dyz,dx2y2,dxy
Lz_d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, -1j, 0.0, 0.0],
                 [0.0, 1j, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, -2j],
                 [0.0, 0.0, 0.0, 2j, 0.0]],dtype=np.complex128)

Sigma_x = np.zeros((2,2),dtype=np.complex128)
Sigma_x[0,1] = 1.0
Sigma_x[1,0] = 1.0
Sigma_y = np.zeros((2,2),dtype=np.complex128)
Sigma_y[0,1] = -1.0j
Sigma_y[1,0] = 1.0j
Sigma_z = np.zeros((2,2),dtype=np.complex128)
Sigma_z[0,0] = 1.0
Sigma_z[1,1] = -1.0

class Atom:
    def __init__(self, name="Atom",position=[0,0,0],orbital_character=[1],hamiltonian_indices = [1],orientation = [1,0,0],SOC_LdotS = [0.0,0.0,0.0],SOC_Rashba = [0.0,0.0,0.0]):
        #name = name of the atom
        #position = position of the atom in the unit cell
        #orbital_character = list of orbital characters s=0,px=1,py=2,pz=3,dz2=4,dxz=5,dyz=6,dx2y2=7,dxy=8
        #hamiltonian_indices = list of the matrix index that this atom corresponds to.
        #orientation = orientation of the x-axis. This is used to rotate the orbitals to the correct orientation. default is 1,0,0, but if you rotate an atom by 45 degrees with respect to the x-lattice direction, then you need to put [1,-1,0].
        #SOC_LdotS = magnitude of the on-site atomic spin orbit coupling. [p_orbitals,d_orbitals,f_orbitals]. 
        self.name = name
        self.position = position
        self.orbital_character = orbital_character
        self.hamiltonian_indices = hamiltonian_indices
        self.orientation = orientation

        #We seperate p,d,f spin orbit strengths on the same atom. But you can also specify a single number for all orbitals for simplicity.
        if not isinstance(SOC_LdotS, list):
            SOC_LdotS = [SOC_LdotS]
        if len(SOC_LdotS) == 1:
            SOC_LdotS = SOC_LdotS*3

        #If SOC_Rash is not a list
        if not isinstance(SOC_Rashba, list):
            SOC_Rashba = [SOC_Rashba]
        if len(SOC_Rashba) == 1:
            SOC_Rashba = SOC_Rashba*3
        
        self.SOC_LdotS = SOC_LdotS
        self.SOC_Rashba = SOC_Rashba


def angle_between_points(point1, point2):
    """Calculate the angle between two points relative to the positive x-axis."""
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def clockwise_order(points, center):
    """Sort the points in clockwise order around the center point."""
    return sorted(points, key=lambda point: angle_between_points(center, point))


@timeit
def Calculate_kpoints_IBZ(Kpoints_BZ,reciprocalLatticeVectors,corners):
    # Create a Polygon object from the corners
    #print("IBZ_Corners",len(corners))

    #first get center of points
    maxcorner = 0
    for i in range(len(corners)):
        corners[i][0] = np.round(corners[i][0],6)
        corners[i][1] = np.round(corners[i][1],6)
        corners[i][2] = np.round(corners[i][2],6)
        cornerLength = np.linalg.norm(np.array(corners)[i])
        if cornerLength > maxcorner:
            maxcorner = cornerLength
    #print("maxcorner",maxcorner)
    

    # Sum of x and y coordinates
    sum_x = sum(corner[0] for corner in corners)
    sum_y = sum(corner[1] for corner in corners)
    sum_z = sum(corner[2] for corner in corners)

    # Calculate the average
    center_x = sum_x / len(corners)
    center_y = sum_y / len(corners)
    center_z = sum_z / len(corners)

    ordered_points = clockwise_order(corners, (center_x,center_y,center_z))
 
    polygon = Polygon(ordered_points)
    minx, miny, maxx, maxy = polygon.bounds
    
    largerKpoints = []
    for point in Kpoints_BZ:
        if minx < point[0] < maxx:
            if miny < point[1] < maxy:
                largerKpoints.append(np.array([point[0],point[1],point[2]]))
    for point in Kpoints_BZ:
        if minx < point[0] +reciprocalLatticeVectors[0][0] < maxx:
            if miny < point[1]+reciprocalLatticeVectors[0][1] < maxy:
                largerKpoints.append(np.array([point[0],point[1],point[2]]) +   np.array(reciprocalLatticeVectors[0]))
    for point in Kpoints_BZ:
        if minx < point[0]  +reciprocalLatticeVectors[1][0]< maxx:
            if miny < point[1] +reciprocalLatticeVectors[1][1]< maxy:
                largerKpoints.append(np.array([point[0],point[1],point[2]]) +   np.array(reciprocalLatticeVectors[1]))
    for point in Kpoints_BZ:
        if minx < point[0]+reciprocalLatticeVectors[2][0] < maxx:
            if miny < point[1] +reciprocalLatticeVectors[2][1]< maxy:
                largerKpoints.append(np.array([point[0],point[1],point[2]]) +   np.array(reciprocalLatticeVectors[2]))
    
    print("largerKpoints", len(largerKpoints))
    """
    #x, y = polygon.exterior.xy

    # Plot the polygon
    plt.plot(x, y)
    plt.fill(x, y, alpha=0.5)  # Fill the polygon
    plt.title('Polygon Shape')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    """
    
    # List of points you want to check
    Kpoints_IBZ = []
    #Kpoints_not_IBZ = []
    #print("HELLO",len(largerKpoints))
    for point in largerKpoints:
        # Create a Point object from each point
        if np.linalg.norm(point) < maxcorner:
            point_obj = Point(point)
            # Check if the point is inside the polygon
            if polygon.contains(point_obj):
                #print(f"Point {point} is inside the polygon.")
                Kpoints_IBZ.append(point)
            elif polygon.touches(point_obj):
                #print("Point touches the edges of the polygon.")
                Kpoints_IBZ.append(point)
    print("Kpoints_IBZ", len(Kpoints_IBZ))
    return Kpoints_IBZ


def calculateBZPlanes(Lattice):
    #Taken from http://lampx.tugraz.at/~hadley/ss1/bzones/drawing_BZ.php. Whowever wrote this i love you.
    G = []#;  //the reciprocal lattice vectors, the fourth index checks if it is in the 1 Bz
    G1 = []#; //the reciprocal lattice vectors used to draw the 1st Bz
    hkl = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[0,1,1],[0,-1,-1],[1,0,0],[-1,0,0],[1,0,1],[-1,0,-1],[1,1,0],[-1,-1,0],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[-1,1,0],[1,-1,0],[1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1]];
    corners = []#; //the first elements are the coordinates, the next three are the planes that intersect at that point, the last is the Bz index.
    cornersbz = []#; //the corners of the Bz zone
    edges = []#;

    A = []#;  //A, b and x are used to find the corners
    b = np.zeros(3,dtype=np.float64)#;
    x = []#;

    #Unit cell primative lattice vectors
    a1x,a1y,a1z = Lattice[0]
    a2x,a2y,a2z = Lattice[1]
    a3x,a3y,a3z = Lattice[2]

    #Recirpocal lattice vectors
    v = a1x*(a2y*a3z-a2z*a3y)+a1y*(a2z*a3x-a2x*a3z)+a1z*(a2x*a3y-a2y*a3x)
    b1x = 2*np.pi*(a2y*a3z-a2z*a3y)/v
    b1y = 2*np.pi*(a2z*a3x-a2x*a3z)/v
    b1z = 2*np.pi*(a2x*a3y-a2y*a3x)/v
    b2x = 2*np.pi*(a3y*a1z-a3z*a1y)/v
    b2y = 2*np.pi*(a3z*a1x-a3x*a1z)/v
    b2z = 2*np.pi*(a3x*a1y-a3y*a1x)/v
    b3x = 2*np.pi*(a1y*a2z-a1z*a2y)/v
    b3y = 2*np.pi*(a1z*a2x-a1x*a2z)/v
    b3z = 2*np.pi*(a1x*a2y-a1y*a2x)/v

    for i in range(len(hkl)):
        G.append([hkl[i][0]*b1x+hkl[i][1]*b2x+hkl[i][2]*b3x,hkl[i][0]*b1y+hkl[i][1]*b2y+hkl[i][2]*b3y,hkl[i][0]*b1z+hkl[i][1]*b2z+hkl[i][2]*b3z,1])

    #print("RECIPROCAL_old")
    #for i in G:
    #    print(i)
    #print("G:",G)
    for i in range(len(hkl)):  #//find the planes that form the boundaries of the first Bz
        Dgamma = np.sqrt(np.power(0.5*G[i][0],2)+np.power(0.5*G[i][1],2)+np.power(0.5*G[i][2],2))  #//distance to Gamma
        for j in range(int((len(hkl)/2))):
            if (j!=i):
                GG = np.sqrt(np.power(G[i][0]/2-G[j][0],2)+np.power(G[i][1]/2-G[j][1],2)+np.power(G[i][2]/2-G[j][2],2))
                if (GG <= Dgamma):
                    G[i][3] = 0 #// this G is not part of the Bz boundary
                    break

    #print("G_updated:",G)
    Np = 0 #number of planes
    for i in range(len(hkl)):
        if (G[i][3] == 1):
            #print(hkl[i][0],hkl[i][1],hkl[i][2])
            G1.append([G[i][0],G[i][1],G[i][2]])
            #print(G1[-1])
            Np+=1
        #print("PLANES_old")
        #for i in G1:
        #    print(i)
        #print(f"There are {Np} planes \n")
        #print(f"There are {Np} planes")
        #print("G1:",G1)
        n = 0
        #print("NUMBER OF PLANES OLD",Np)
    return G,G1

def calculateBZCorners(G,G1):
    #Taken from http://lampx.tugraz.at/~hadley/ss1/bzones/drawing_BZ.php. Whowever wrote this i love you.
    hkl = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[0,1,1],[0,-1,-1],[1,0,0],[-1,0,0],[1,0,1],[-1,0,-1],[1,1,0],[-1,-1,0],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[-1,1,0],[1,-1,0],[1,0,-1],[-1,0,1],[0,1,-1],[0,-1,1]];
    corners = []#; //the first elements are the coordinates, the next three are the planes that intersect at that point, the last is the Bz index.
    cornersbz = []#; //the corners of the Bz zone
    edges = []#;

    A = []#;  //A, b and x are used to find the corners
    b = np.zeros(3,dtype=np.float64)#;
    x = []#;
    n = 0
    for i in range(2,len(G1)):
        for j in range(1,i):
            for k in range(0,j):
                A = np.array([[G1[i][0], G1[i][1], G1[i][2]],[G1[j][0], G1[j][1], G1[j][2]],[G1[k][0], G1[k][1], G1[k][2]]]);
                b[0] = (G1[i][0]*G1[i][0]+G1[i][1]*G1[i][1]+G1[i][2]*G1[i][2])/2;
                b[1] = (G1[j][0]*G1[j][0]+G1[j][1]*G1[j][1]+G1[j][2]*G1[j][2])/2;
                b[2] = (G1[k][0]*G1[k][0]+G1[k][1]*G1[k][1]+G1[k][2]*G1[k][2])/2;
                #print("old",i,j,k,A)
                if (np.linalg.det(A)!=0):
                    x = np.linalg.solve(A, b)
                    corners.append([x[0],x[1],x[2],1])
                    n+=1


    for i in range(n):#(i=0; i<n; i++)# {  //find the corners of the Bz by choosing only the corners closer to Gamma than another G
        Dgamma =np.sqrt(np.power(corners[i][0],2)+np.power(corners[i][1],2)+np.power(corners[i][2],2)) # //distance to Gamma
        for j in range(len(hkl)):#(j=0; j<26; j++) {
            if (j!=i): #{
                DGG = np.sqrt(np.power(corners[i][0]-G[j][0],2)+np.power(corners[i][1]-G[j][1],2)+np.power(corners[i][2]-G[j][2],2))
                if (DGG < Dgamma):
                  corners[i][3] = 0#; // this corner is not part of the Bz boundary
                  break

    #print("corners updated",corners)
    for i in range(1,n):#(i=1; i<n; i++) {  //check to see if the corners are unique
        for j in range(0,i):#(j=0; j<i; j++) {
          if ((corners[i][0]==corners[j][0])&(corners[i][1]==corners[j][1])&(corners[i][2]==corners[j][2])):# {
            corners[j][3]=0
    #print("corners second updated",corners)

    Nc = 0
    dmax = 0
    for i in range(n):#(i=0; i<n; i++) {
        if (corners[i][3] == 1):
            cornersbz.append([corners[i][0],corners[i][1],corners[i][2]])
            d = np.round(np.sqrt(corners[i][0]*corners[i][0]+corners[i][1]*corners[i][1]+corners[i][2]*corners[i][2]),4)
            dmax = np.max([dmax,d])
            Nc+=1

    return cornersbz

def calculateBZEdges(G1, cornersbz):
    edges = []
    Ne = 0
    for i in range(1,len(cornersbz)):#(i=1; i<Nc; i++) {  //for every pair of corners check every pair of planes to see if the corners both lie in those planes
        for j in range(i):#(j=0; j<i; j++) {
            for k in range(len(G1)):#(k=1; k<Np; k++) {
                for l in range(k):# (l=0; l<k; l++) {
                    d_ki = G1[k][0]*cornersbz[i][0]+G1[k][1]*cornersbz[i][1]+G1[k][2]*cornersbz[i][2]-(G1[k][0]*G1[k][0]+G1[k][1]*G1[k][1]+G1[k][2]*G1[k][2])/2
                    d_li = G1[l][0]*cornersbz[i][0]+G1[l][1]*cornersbz[i][1]+G1[l][2]*cornersbz[i][2]-(G1[l][0]*G1[l][0]+G1[l][1]*G1[l][1]+G1[l][2]*G1[l][2])/2
                    d_kj = G1[k][0]*cornersbz[j][0]+G1[k][1]*cornersbz[j][1]+G1[k][2]*cornersbz[j][2]-(G1[k][0]*G1[k][0]+G1[k][1]*G1[k][1]+G1[k][2]*G1[k][2])/2
                    d_lj = G1[l][0]*cornersbz[j][0]+G1[l][1]*cornersbz[j][1]+G1[l][2]*cornersbz[j][2]-(G1[l][0]*G1[l][0]+G1[l][1]*G1[l][1]+G1[l][2]*G1[l][2])/2
                    delta = 1e-20
                    if ((np.abs(d_ki)<delta)&(np.abs(d_li)<delta)&(np.abs(d_kj)<delta)&(np.abs(d_lj)<delta)):
                        edges.append([cornersbz[i][0],cornersbz[i][1],cornersbz[i][2],cornersbz[j][0],cornersbz[j][1],cornersbz[j][2]])
                        #print(edges[-1][0],edges[-1][1],edges[-1][2])
                        Ne+=1
                        break
    
    return edges

def calculateBZFaces(G1, BZcorners):
        faces = []
        for k in range(len(G1)):
            face = []
            for i in range(len(G1)):
                d_ki = G1[k][0]*BZcorners[i][0] + G1[k][1]*BZcorners[i][1] + G1[k][2]*BZcorners[i][2] - (G1[k][0]*G1[k][0] + G1[k][1]*G1[k][1] + G1[k][2]*G1[k][2])/2.0
                delta = 1e-4
                if abs(d_ki) < delta:
                    facepoint = [BZcorners[i][0], BZcorners[i][1], BZcorners[i][2]]
                    face.append(facepoint)
            faces.append(face)
        return faces
