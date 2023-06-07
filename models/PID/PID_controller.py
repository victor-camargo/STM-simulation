from plant.stm_dynamics import STM_sim
import numpy as np

def PID(Kp, Ki,  Kd, delta_T = 1e-3, MV_bar=0):
    # initialize stored data
    e_prev = 0
    delta_T = delta_T
    I = 0

    # initial control
    MV = MV_bar
    

    while True:
        
        # yield MV, wait for new t, PV, SP
        PV, SP = yield MV

        # PID calculations
        e = SP - PV

        P = Kp*e
        I = I + Ki*e*delta_T
        D = Kd*(e - e_prev)/delta_T

        MV = MV_bar + P + I + D

        # update stored data for next iteration
        e_prev = e


class PIDControlSTM:
    def __init__(self, stm_model, Kp=1, Ki=1, Kd=1, MV_bar=0):
    
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.MV_bar = MV_bar
        
        
        self.stm_model = stm_model
        
        self.model = PID(self.Kp, self.Ki, self.Kd, self.stm_model.dT, self.MV_bar)
        self.model.send(None) 
    

    
    def controller(self, perturbation, desired_out):

            u = 0
            results = []
            
            for point in perturbation:
                applied_u, current, delta_z = self.stm_model.simulate(u , point)                
                results.append([applied_u, current, delta_z])
                u = self.model.send([current, desired_out])

            return np.array(results).T
