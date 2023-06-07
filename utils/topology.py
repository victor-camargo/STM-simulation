import numpy as np

# Cria semicirculos simulando átomos não interagentes no estado 1s sem

def atom_func(x, R):
    return np.sqrt(R**2-(x-2*R*np.floor((x/(2*R))+0.5))**2)
