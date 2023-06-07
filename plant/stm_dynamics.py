import numpy as np
from scipy.integrate import odeint


class STM_sim:
    def __init__(self,
                 sigma=10,
                 work_fn=2,
                 Vb=10,
                 offset=10,
                 u=0,
                 z0=0,
                 v0=0,
                 x0=0,
                 t=0,
                 dT=1e-3):
        
        self.sigma = sigma
        self.work_fn = work_fn
        self.Vb = Vb
        self.offset = offset
        
        self.m = 1e-3 # Massa do piezo
        self.b_ = 1e-1 # Coeficiente de viscosidade
        self.a = 0.8 # Coeficiente de pós-escoamento
        self.ki = 1e-3 # rigidez

        # Constantes para medir o deslocamento inerente à histerese
        self.alpha = 1 
        self.beta = 0.1
        self.gamma = 0.9
        self.n = 2
        
        self.u = u # Sinal de entrada para o piezo
        self.z0 = z0
        self.v0 = v0
        self.x0 = x0
        self.t = t
        self.dT = dT
    
   
    def aparent_barrier(self, delta_z):
        return self.sigma*self.Vb*np.exp(-1.025*np.sqrt(self.work_fn)*delta_z)

    def plant_simulation(self, z, h):
        
        # A distância da ponta de prova até a amostra é dada pela seguinte equação, onde h é a topologia da amostra
        delta_z = self.offset - z -h
        
        return self.aparent_barrier(delta_z), delta_z
    
    # Equação de Bouc-Wen para modelagem do deslocamento do piezoelétrico
    def piezo_bouc_wen_equation(self, x, t):
        dx_dt = [0, 0, 0]

        dx_dt[0] = x[1]
        dx_dt[1] = -(self.b_/self.m)*x[1] -((self.a*self.ki)/self.m)*x[0] - (((1-self.a)*self.ki)/self.m)*x[2] + (1/self.m)*self.u
        dx_dt[2] = x[1]*(self.alpha -(self.beta*np.sign(x[2]*x[1])+self.gamma)*np.abs(x[2])**self.n)
        return dx_dt

    def z_dynamic(self, z0, v0, x0, t, dT):
        #print("u:{}\nz:{}\nv:{}\nx:{}\nt:{}\ndT:{}".format(self.u, z0, v0, x0, t, dT))
        z, v, histeretic_displacement = odeint(self.piezo_bouc_wen_equation, y0=[z0, v0, x0], t=[t,t+ dT])[-1]

        return z, v, histeretic_displacement

    def set_input(self, u:float):
        #saturação da entrada
        self.u = np.maximum(-50, np.minimum(u,50))
        
    def reset_time(self):
        self.t = 0

    def simulate(self, u:float, topology:float)->float:
        
        self.set_input(u)
        
        z,v,x = self.z_dynamic(self.z0, self.v0, self.x0, self.t, self.dT)
        self.z0 = z
        self.v0 = v
        self.x0 = x

        self.t = self.t + self.dT
        
        current, delta_z = self.plant_simulation(z, topology)
        return self.u, current, delta_z
            
            
            
