from plant.stm_dynamics import STM_sim
import numpy as np

class PIDNNControl:
    def __init__(self,
                 stm_model,
                 hurwitz_constant = -3,
                 damping_factor = 1e-2,
                 learning_rate = 0.5):

        self.fc = np.ones((1,3))
        self.a = np.zeros((1,3))
        self.errors = np.zeros((1,3))
        
        self.hurwitz_constant = hurwitz_constant
        self.learning_rate = learning_rate
        self.damping_factor = damping_factor
                
        self.stm_model = stm_model



    def forward(self, x):

        self.errors[1:] = self.errors[:-1]
        self.errors[0] = x # a entrada é o erro

        self.a[0,0] = self.fc[0,0]*self.errors[0,0] # Parte proporcional
        self.a[0,1] = self.a[0,1] + self.fc[0,1]*self.errors[0,0]*self.stm_model.dT
        self.a[0,2] = self.fc[0,2]*(self.errors[0,0] - self.errors[0,1])/self.stm_model.dT
        
        return np.sum(self.a)
    
    
    def backward(self, y_pred, y_ref):
        feed = self.forward(y_ref - y_pred)
        
        #loss
        
        self.loss = 0.5*(y_pred - y_ref)**2 
        
        # Backward pass
        delta_y_pred = y_ref - y_pred
        delta_u = delta_y_pred*(1/self.hurwitz_constant)
        delta_fc = delta_u*np.divide(self.a, self.fc)

        self.fc -= (self.learning_rate * delta_fc + self.damping_factor*abs(y_pred - y_ref)*self.fc)


    def controller(self,  perturbation, desired_out):

        # online learning
        u = 0
        
        alpha = self.learning_rate
        eta = self.damping_factor

        results = []
        for k, point in enumerate(perturbation):
            
            #self.learning_rate = alpha*np.exp(-(k/100)*np.log(10))
            #self.damping_factor = eta*np.exp((t/1e5)*np.log(2))

            applied_u, current, delta_z = self.stm_model.simulate(u , point)                
            results.append([u, current, delta_z])

            print(self.fc)
            
            loss = self.backward( current, desired_out)

            u = self.forward(desired_out - current)


        return np.array(results).T


