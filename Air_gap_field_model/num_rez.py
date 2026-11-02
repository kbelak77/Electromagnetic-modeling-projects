import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


data_1 = np.loadtxt("By.txt", skiprows=7)
data_2 = np.loadtxt("Bx.txt", skiprows=7)
data_3 = np.loadtxt("By_utori.txt", skiprows=7)
data_4 = np.loadtxt("Bx_utori.txt", skiprows=7)

x_1 = data_1[:, 0] 
By1 = data_1[:, 1]

x_2 = data_2[:, 0] 
Bx2 = data_2[:, 1]

x_3 = data_3[:, 0] 
By3 = data_3[:, 1]

x_4 = data_4[:, 0] 
Bx4 = data_4[:, 1]


plt.figure()
plt.plot(x_1, By1, label=r"$B_y$ ", color = 'orange')
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{y}$')
plt.legend(loc ='upper right')
plt.grid(True)
plt.savefig("By_bezut.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(x_2, Bx2, label=r"$B_x$ ", color = 'orange')
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{x}$')
plt.legend()
plt.grid(True)
plt.savefig("Bx_bezut.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(x_3, By3, label=r"$B_y$", color = 'orange')
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{y}$')
plt.legend()
plt.grid(True)
plt.savefig("Graf_By.pdf", bbox_inches="tight") 

plt.figure()
plt.plot(x_4, Bx4, label=r"$B_x$", color = 'orange')
plt.xlabel('opseg [mm]')
plt.ylabel(r'$B_{x}$')
plt.legend()
plt.grid(True)
plt.savefig("Graf_Bx.pdf", bbox_inches="tight") 

if __name__ == "__main__":
    plt.show()