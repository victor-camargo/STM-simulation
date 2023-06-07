from plant.stm_dynamics import STM_sim
from models.PID.PID_controller import *
from models.AdaptNN.NN import *
from models.PIDNeural.PIDNN_controller import *
from utils.topology import atom_func
import matplotlib.pyplot as plt
import numpy as np

simulator = STM_sim()
x = np.linspace(0,100,1000)

atom_surface = atom_func(x, 0.5)

#plt.plot(x, atom_surface)
#plt.show()


#atom_surface = np.ones(x.shape[-1])

#PID_model = PIDControlSTM(simulator, Kp=50, Ki=5, Kd=0.2)
#results = PID_model.controller(atom_surface, 3)

'''
NN_model = NeuralControl(simulator,
                         hurwitz_constant=500,
                         damping_factor=1e-3,
                         learning_rate = 0.5)
'''
#results = NN_model.controller(atom_surface, 3)

PIDNN_model = PIDNNControl(simulator,
                         hurwitz_constant=-10,
                         damping_factor=1e-4,
                         learning_rate = 0.5)
results = PIDNN_model.controller(atom_surface, 3)

print(results[1])
plt.plot(x, atom_surface, x, atom_surface + results[2])
plt.show()
