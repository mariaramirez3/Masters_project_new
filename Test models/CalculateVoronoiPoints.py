import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
#from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d


#import matplotlib.tri as tri
#from matplotlib.colors import ListedColormap

#change to directory of current script
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_data(file_name):
    #This is specifically for a data file i made for this calculation. not standardised.
    f = open(file_name, 'r')
    data = []
    for line in f:
        l = line.split()
        #exit(0)
        file= l[0].split("/")[-1].split("_") #first entry should be 'Sr2RuO4_U_J_nk_nkf.out'
        U = float(file[1])
        J = float(file[2])
        nk = int(file[3])
        nkf = int(file[4].split(".")[0])

        time= l[3].split("m")
        seconds = float(time[0])*60 + float(time[1].split("s")[0])
        final_data = [U, J, nk, nkf,l[1],float(l[2]),seconds] 
        data.append(final_data)

    return data

def plot_U_J_Voronoi(data,inputname,outputname):
    print(inputname,outputname,"PYTHON",os.getcwd())
    # Triangular grid
    kb_ev = 8.6173303e-5 #eV
    x = []
    y = []
    groundstate = []
    points = []
    z = []
    for d in data:
       if d[0] < 3.05:
           if d[1] < 1.05:
                x.append(d[0])
                y.append(d[1])
                points.append([d[0],d[1]])
                groundstate.append(d[4])
                z.append((d[5])/kb_ev)
    
    #convert ground_State to numbers
    groundstate_num = []
    for i in range(len(groundstate)):
        if groundstate[i] == "FL":
            groundstate_num.append(0)
        elif groundstate[i] == "SC":
            groundstate_num.append(1)
        elif groundstate[i] == "SDW":
            groundstate_num.append(2)
        elif groundstate[i] == "CDW":
            groundstate_num.append(3)
        else:
            groundstate_num.append(-1)

    max_U = max(x)
    min_U = min(x)
    max_J = max(y)
    min_J = min(y)

    #Add in boundaries
    for i in np.linspace(min_U,max_U,31):
        points.append([i,-1])
        groundstate.append("Empty")
        groundstate_num.append(-1) 
    
    for i in np.linspace(min_J,max_J,31):
        points.append([-1,i])
        groundstate.append("Empty")
        groundstate_num.append(-1)


    points = np.array(points)
    categories = np.array(groundstate_num)  # Random categories (0, 1, 2, 3)
    
    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Plot Voronoi diagram
        # Set background colors
    #fig, ax = plt.subplots(figsize=(9, 3))
    #fig.patch.set_facecolor('black')  # Set figure background
    #ax.set_facecolor('white')         # Set axes background
    #voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=0.0, show_points=False)

    # Highlight points with categories
    for i in range(len(x)):
        if groundstate[i] == "SC":
            colour = 'blue'
        elif groundstate[i] == "CDW":
            colour = 'green'
        elif groundstate[i] == "FL":
            colour="black"
        elif groundstate[i] == "SDW":
            colour="orange"
        else:
            colour="white"
        #ax.scatter(x[i], y[i], c=colour, s=20,marker="x")


    # Identify boundaries between different categories
    f = open(outputname,"w")

    lines = []
    line_magnitudes = []
    # Identify boundaries between different categories, calculate their magnitudes
    for ridge, (p1, p2) in zip(vor.ridge_vertices, vor.ridge_points):
        if ridge[0] == -1 or ridge[1] == -1:  # Skip infinite ridges
            continue
        if categories[p1] != categories[p2]:  # Different categories
            line = vor.vertices[ridge] ##[[x_start,y_start],[x_end,y_end]]
            lines.append(line)
            line_magnitudes.append(np.sqrt((line[1][0]-line[0][0])**2 + (line[1][1]-line[0][1])**2))
            #ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=2)
    
    # Sort lines by their length
    sorted_lines = sorted(lines, key=lambda line: ((line[1][0] - line[0][0])**2 + (line[1][1] - line[0][1])**2)**0.5)
    sorted_lines = sorted_lines[::-1] #this puts longest lines first

    filtered_lines = []
    for line in sorted_lines:
        if not (line[0][0] < 0.0 or line[0][1] < 0.0 or line[1][0] < 0.0 or line[1][1] < 0.0):
            filtered_lines.append(line)

    
    #print(len(sorted_lines))
    #save the start, end and mid point of the Voronoi line.
    for i in range(int(len(filtered_lines)/5)): #take 10% of the longest lines and write those into the list. 
            x = (filtered_lines[i][0, 0] + filtered_lines[i][1, 0])/2.0
            y = (filtered_lines[i][0, 1] + filtered_lines[i][1, 1])/2.0
            f.write(f"{np.round(x,3)} {np.round(y,3)}\n")
	    #if x > 0 and y > 0:
            #    f.write(f"{np.round(x,3)} {np.round(y,3)}\n")
            #x = sorted_lines[i][0,0]
            #y = sorted_lines[i][0,1]
            #if x > 0 and y > 0:
            #    f.write(f"{np.round(x,3)} {np.round(y,3)}\n")
            #x = sorted_lines[i][1,0]
            #y = sorted_lines[i][1,1]
            #if x > 0 and y > 0:
            #    f.write(f"{np.round(x,3)} {np.round(y,3)}\n")
    f.close()

    # Plot settings
    #ax.set_title(f"{inputname}")
    #ax.set_xlim(-0.05, 3.05)
    #ax.set_ylim(-0.05, 1.05)
    #plt.xlabel("U /eV")
    #plt.ylabel("J /eV")

    #plt.savefig(f"{inputname}.png")
    #plt.savefig(f"{inputname}.pdf")
    #plt.show()

def main():
    
    inputname = sys.argv[1]
    outputname = sys.argv[2]
    data = read_data(inputname)
    plot_U_J_Voronoi(data,inputname,outputname)

if __name__ == "__main__":
    main()



