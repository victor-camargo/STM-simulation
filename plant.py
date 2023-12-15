import torch
import torch.nn as nn
import numpy as np


class STMFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, u, h, x, hurwitz_constant, dt, params):  # estado da ponta de prova, sinal de entrada, altura da amostra, passo, parâmetros de configuração

        self.hurwitz_constant = hurwitz_constant
        dx_dt = torch.zeros(3)

        dx_dt[0] = x[1]
        dx_dt[1] =  (1/params['m'])*(params['d']*u -params['b_']*x[1] -params['a']*params['ki']*x[0] - (1-params['a'])*params['ki']*x[2])
        dx_dt[2] = x[1]*(params['alpha'] -(params['beta']*torch.sign(x[2]*x[1])+params['gamma'])*torch.abs(x[2])**params['n'])

        new_x = x + dt*dx_dt #  Método de Euler

        delta_z = torch.tensor([params['offset']]) - new_x[0] - h # distância da ponta de prova até a superfície

        current = params['sigma']*params['Vb']*torch.exp(-1.025*torch.sqrt(torch.tensor([params['work_fn']]))*delta_z)

        linearized_current = torch.log(current)

        return linearized_current, delta_z, new_x

    @staticmethod
    def backward(self, grad_current, grad_delta_z, grad_new_x):
        return -grad_current*self.hurwitz_constant**(-1), None, None, None, None, None


plant_params = {
                'sigma': 1,  # Coeficiente de proporcionalidade da corrente de tunelamento
                'work_fn': 2,  # Função trabalho
                'Vb': 10,  # Tensão de polarização
                'offset': 5,  # distancia do piezo com relação a menor altura
                'm': 1,  # Massa do piezo
                'b_': 1,  # Coeficiente de viscosidade
                'a': .95,  # Coeficiente de pós-escoamento
                'ki': 1,  # rigidez
                'd': 1,  # Coeficiente do piezo m/V

                # Constantes para medir o deslocamento inerente à histerese
                'alpha': 1, 
                'beta': 0.8,
                'gamma': 0.2,
                'n':  2, }

class STMPlant(nn.Module):

    def __init__(self,
                x0 = torch.zeros(3, requires_grad=False),
                hurwitz_constant = -1,
                plant_params = plant_params):
        
        super(STMPlant, self).__init__()
        
        self.params = plant_params
        
        self.plant = STMFunction.apply

        self.hurwitz_constant = hurwitz_constant

        self.x0 = x0

    def forward(self, u, h, dT):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        current, delta_z, new_x0 = self.plant( u, h, self.x0, self.hurwitz_constant, dT, self.params)
        self.x0.data = new_x0.data
        return current, delta_z, new_x0


def stm_sim(u, h, x, dt, params): # estado da ponta de prova, sinal de entrada, altura da amostra, passo, parâmetros de configuração
    
    dx_dt = np.zeros(3)

    dx_dt[0] = x[1]
    dx_dt[1] =  (1/params['m'])*(params['d']*u -params['b_']*x[1] -params['a']*params['ki']*x[0] - (1-params['a'])*params['ki']*x[2])
    dx_dt[2] = x[1]*(params['alpha'] -(params['beta']*np.sign(x[2]*x[1])+params['gamma'])*np.abs(x[2])**params['n'])

    new_x = x + dt*dx_dt # Método de Euler 
    
    delta_z = params['offset'] - new_x[0] - h # distância da ponta de prova até a superfície

    current = params['sigma']*params['Vb']*np.exp(-1.025*np.sqrt(params['work_fn'])*delta_z)

    linearized_current = np.log(current)

    return linearized_current, delta_z, new_x


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        

    def forward(self, y_true, y_pred):
      # Calculate the euclidian distance and calculate the contrastive loss

      loss_contrastive = torch.pow(y_true - y_pred, 2)


      return loss_contrastive



