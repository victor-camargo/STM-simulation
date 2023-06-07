from plant.stm_dynamics import STM_sim
import numpy as np

class NeuralControl:
    def __init__(self,
                 stm_model,
                 input_size= 20,
                 hidden_size=[20, 20, 20],
                 output_size = 1,
                 activation_fn = ['relu', 'tanh', 'tanh'],
                 hurwitz_constant = -3,
                 damping_factor = 1e-2,
                 learning_rate = 0.5):
        
        #'relu', 'relu','relu','relu'
        #'sigmoid', 'sigmoid','sigmoid', 'sigmoid'
                
        self.activation_fn = activation_fn
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
                
        self.fc = []
        
        self.fc.append(self.xavier_initialization(self.input_size, self.hidden_size[0])) # camada de entrada
       
        for i in range(len(hidden_size)-1):
            self.fc.append(self.xavier_initialization(self.hidden_size[i], self.hidden_size[i+1]))
        
        self.fc.append(self.xavier_initialization(self.hidden_size[-1], self.output_size))
        
        self.n_layers = len(self.fc)
        
        self.b = [0]*self.n_layers
        self.z = [0]*self.n_layers
        self.a = [0]*self.n_layers
        
        
        self.delta_z = [0]*self.n_layers
        self.delta_a = [0]*self.n_layers
        self.delta_fc = [0]*self.n_layers
        self.delta_b = [0]*self.n_layers
        
        
        self.output_activation = 'lin'

        
        self.hurwitz_constant = hurwitz_constant
        self.learning_rate = learning_rate
        self.damping_factor = damping_factor
                
        self.stm_model = stm_model
        
    def xavier_initialization(self, previous_layer, next_layer):
        return np.random.randn(previous_layer, next_layer)*np.sqrt(1./previous_layer)
        #return 2*np.random.random((previous_layer, next_layer)) -1
    
    def zeros_initialization(self, previous_layer, next_layer):
        return np.zeros((previous_layer, next_layer))
    # Funções de ativação
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def lin(self, x):
        return x
    
    # Derivadas das funções de ativação
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)
    
    def tanh_derivative(self,x):
        return (1/np.cosh(x))**2
    
    def lin_derivative(self, x):
        return 1
    
    
    
    def activation_function(self, x, fn):
        if fn == 'relu':
            return self.relu(x)
        elif fn == 'sigmoid':
            return self.sigmoid(x)
        elif fn == 'tanh':
            return self.tanh(x)
        elif fn == 'lin':
            return self.lin(x)
        
    def d_activation_function(self, x, fn):
        if fn == 'relu':
            return self.relu_derivative(x)
        elif fn == 'sigmoid':
            return self.sigmoid_derivative(x)        
        elif fn == 'tanh':
            return self.tanh_derivative(x)
        elif fn == 'lin':
            return self.lin_derivative(x)
   

    def forward(self, x):
        self.z[0] = np.dot(x, self.fc[0]) + self.b[0]
        self.a[0] = self.activation_function(self.z[0], self.activation_fn[0])
        
        for i in range(1, self.n_layers-1):
            self.z[i] = np.dot(self.a[i-1], self.fc[i]) + self.b[i]
            self.a[i] = self.activation_function(self.z[i], self.activation_fn[i])
    
        

        self.z[-1] = np.dot(self.a[-2], self.fc[-1]) + self.b[-1]
        self.a[-1] = self.activation_function(self.z[-1], self.output_activation)
    

        return self.a[-1]
    
    
    def backward(self, x, y_pred, y_ref):
        feed = self.forward(x)
        
        #loss
        
        self.loss = 0.5*(y_pred - y_ref)**2 
        
        # Backward pass
        delta_y_pred = y_pred - y_ref
        delta_u = delta_y_pred*(1/self.hurwitz_constant)#*np.sign((x[0,2] - x[0,1])/(feed - x[0, 250] + 1e-10))
        self.delta_a[-1] = delta_u
        self.delta_z[-1] = self.delta_a[-1]*self.d_activation_function(self.z[-1],self.output_activation)
        self.delta_fc[-1] = np.dot(self.a[-2].T, self.delta_z[-1])
        
        for i in range(self.n_layers-2, 0 , -1):
            self.delta_a[i] = np.dot(self.delta_z[i+1], self.fc[i+1].T)
            self.delta_z[i] = self.delta_a[i]*self.d_activation_function(self.z[i],self.activation_fn[i])
            self.delta_fc[i] = np.dot(self.a[i-1].T, self.delta_z[i])           

        
        self.delta_a[0] = np.dot(self.delta_z[1], self.fc[1].T)
        self.delta_z[0] = self.delta_a[0]*self.d_activation_function(self.z[0],self.activation_fn[0])
        self.delta_fc[0] = np.dot(x.T, self.delta_z[0])     
        
        for i in range(self.n_layers):
            self.fc[i] -= (self.learning_rate * self.delta_fc[i] + self.damping_factor*abs(y_pred - y_ref)*self.fc[i])
            self.b[i] -= (self.learning_rate * self.delta_b[i] + self.damping_factor*abs(y_pred - y_ref)*self.b[i])

        return self.loss

    def controller(self,  perturbation, desired_out):

        # online learning
        u = 0
        X = np.zeros((1,self.input_size))
        
        alpha = self.learning_rate
        eta = self.damping_factor

        results = []
        for point in perturbation:
            
            #self.learning_rate = alpha*np.exp(-(t/1e4)*np.log(2))
            #self.damping_factor = eta*np.exp((t/1e5)*np.log(2))

            applied_u, current, delta_z = self.stm_model.simulate(u , point)                
            results.append([u, current, delta_z])

            
            X[0,1:] = X[0,:-1]
            X[0,0] = desired_out - current

            X[0,10] = applied_u
            print(X)
            
            loss = self.backward(X, current, desired_out)

            u = self.forward(X)[0,0]


        return np.array(results).T


