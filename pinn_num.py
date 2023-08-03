import tensorflow as tf 
import numpy as np 
import scipy.optimize
import matplotlib.pyplot as plt 

DTYPE='float32'

def get_model(input_dim,output_dim,layers): 
    input = tf.keras.Input(shape = (input_dim,))
    x = input
    for n_units in layers : 
        x = tf.keras.layers.Dense(n_units, activation = 'tanh')(x)
    x = tf.keras.layers.Dense(output_dim,activation = 'linear')(x)
    return tf.keras.Model(inputs = input, outputs = x)



class NN_Model(tf.keras.Model): 
    '''
    Neural network model. 
    Tensorflow Model class 
    '''

    def __init__(self,output_dim,layers = [33, 33, 33],**kwargs):
        '''
        Build a Multilayer perceptron 
        Inputs : 
        output_dim ::: int ::: Model output dimension 
        layers ::: list of int ::: number of units in each layer of the MLP 
        Outputs : 
        None 
        '''
        super().__init__(**kwargs)
        self.output_layer = tf.keras.layers.Dense(output_dim)
        self.hidden_layers = [tf.keras.layers.Dense(n_units, 
                                                    activation = 'tanh')
                               for n_units in layers]
    
    def call(self,input):
        '''
        Model call method. Renders the model prediction
        Inputs : 
        input ::: array like object (n_batch, output_dim) ::: input of 
        the model 
        Outputs :
        output ::: array like object (n_batch, output_dim)::: output of 
        the model
        '''
        x = input
        for layer in self.hidden_layers : 
            x = layer(x)
        return self.output_layer(x)

class PINNSolver():
    '''
    PINNSolver object
    '''
    def __init__(self, model, X_r, pde_residual_scaling = 1e-4):
        '''
        Inputs :
        model ::: NN_Model instance ::: Neural network used for training
        X_r ::: array like object ::: collocation points 
        Outputs : 
        None
        '''
        # Initialize model and build model 
        self.model = model
        self.input_dim = np.shape(X_r)[1]
        #self.model.build(input_shape=(self.input_dim,))
        # Initialize collocation points 
        self.collocation = tf.convert_to_tensor(X_r)
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        # Set residual loss scaling 
        self.pde_residual_scaling_ = pde_residual_scaling
    
    def loss_fn(self, X, u):
        ''' 
        Calculates loos function, which is the sum of the PDE loss 
        (viz. residual, calculated on collocation points) with the 
        vanilla loss, here RMSE
        Inputs : 
        X ::: array object (n_batch, input_dim) ::: Input features 
        u ::: array like object (n_batch, output_dim) ::: Target values 
        Output :
        loss ::: float ::: total loss 
        '''
        # Compute PDE residual
        r = self.get_r()
        phi_r = self.pde_residual_scaling_*tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r
        u_pred = self.model(X)
        loss = phi_r + tf.reduce_mean(tf.keras.losses.mean_squared_error(u, u_pred))

        # Add phi_0 and phi_b to the loss
        #for i in range(len(X)):
        #    u_pred = self.model(X[i])
        #    loss += tf.reduce_mean(tf.square(u[i] - u_pred))
        
        return loss
    
    def get_grad(self, X, u):
        '''
        Calculate Loss gradients with respect to model trainable parameters 
        Inputs : 
        X ::: array like object (n_batch,input_dim) ::: input features of 
        the model 
        u ::: array like object (n_batch, output_dim) ::: target values 
        associated with input features 
        Outputs : 
        loss ::: ::: Loss value for the current training step 
        g ::: ::: gradients of the loss with respect to the different parameters 
        '''
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(X, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    
    def solve_with_TFoptimizer(self, optimizer, X, u, N=1001):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(X, u)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        for i in range(N):
            
            loss = train_step()
            
            self.current_loss = loss.numpy()
            self.callback()

    def solve_with_ScipyOptimizer(self, X, u, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, grad = self.get_grad(X, u)

            #print(grad)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss            
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        print('run scipy optimizer')
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        
    def callback(self, xr=None):
        if self.iter % 50 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.iter+=1
        
    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax

class StatNS_PINN(PINNSolver): 

    def __init__(self,model,X_r, viscosity,pde_residual_scaling = 1e-4):
        '''
        '''
        super().__init__(model,X_r,pde_residual_scaling = pde_residual_scaling)
        self.x = X_r[:,0:1]
        self.x = tf.convert_to_tensor(self.x)
        self.y = X_r[:,1:2]
        self.y = tf.convert_to_tensor(self.y)
        self.viscosity = viscosity

    def fun_r(self,u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy):

        res1 = u*u_x + v*u_y + p_x - self.viscosity*(u_xx + u_yy)
        res2 = u*v_x + v*v_y + p_y - self.viscosity*(v_xx + v_yy)
        res3 = u_x + v_y

        returned_val = tf.square(res1)+ tf.square(res2) #tf.reduce_sum(tf.square(res1)) + tf.reduce_sum(tf.square(res2))

        return returned_val
    
    def get_r(self):
        x = self.x
        y = self.y
        with tf.GradientTape(persistent = True) as tape2 :
            tape2.watch(x)
            tape2.watch(y)
            with tf.GradientTape(persistent = True) as tape :
                tape.watch(x)
                tape.watch(y)
                stack = tf.stack((x,y),axis = 1)
                pred = self.model(stack)
                u, v, p = pred[:,0], pred[:,1], pred[:,2]
                p_x = tape.gradient(p,x)
                p_y = tape.gradient(p,y)

                u_x = tape.gradient(u,x)
                u_y = tape.gradient(u,y)
                v_x = tape.gradient(v,x)
                v_y = tape.gradient(v,y)

            u_xx = tape2.gradient(u_x,x)
            u_yy = tape2.gradient(u_y,y)
            v_xx = tape2.gradient(v_x,x)
            v_yy = tape2.gradient(v_y,y)
        
        residual = self.fun_r(u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy)

        del tape, tape2

        return residual
    
    def loss_fn(self, X, Y):
        ''' 
        Calculates loos function, which is the sum of the PDE loss 
        (viz. residual, calculated on collocation points) with the 
        vanilla loss, here RMSE
        Inputs : 
        X ::: array object (n_batch, input_dim) ::: Input features 
        Y ::: array like object (n_batch, output_dim) ::: Target values 
        Output :
        loss ::: float ::: total loss 
        '''
        # Compute PDE residual
        r = self.get_r()
        phi_r = self.pde_residual_scaling_*tf.reduce_sum(r)

        # Add phi_0 and phi_b to the loss
        #for i in range(len(X)):
        #    u_pred = self.model(X[i])
        #    loss += tf.reduce_mean(tf.square(u[i] - u_pred))

        pred = self.model(X)
        u, v, p = pred[:,0], pred[:,1], pred[:,2]
        uv_hat = tf.squeeze(tf.stack((u,v),axis = 1))
 
        loss = phi_r + tf.reduce_sum(tf.square(Y - uv_hat))

        return loss


class HarmOsci_PINN(PINNSolver):

    def __init__(self,model,X_r, mass, viscosity, stiffness): 
        super().__init__(model,X_r)
        self.x = X_r[:,0:1]
        self.x = tf.convert_to_tensor(self.x)
        self.mass = mass 
        self.viscosity = viscosity
        self.stiffness = stiffness

    def fun_r(self, x, u, u_x, u_xx):
        """Residual of the PDE"""
        return self.mass * u_xx + self.viscosity * u_x + self.stiffness * u
    
    def get_r(self):
        # A tf.GradientTape is used to compute derivatives in TensorFlow
        with tf.GradientTape(persistent=True) as tape:

            # Variable x is watched during tape
            # to compute derivatives u_x and u_xx
            tape.watch(self.x)

            # Determine residual
            #u = model(tf.stack([t[:,0], x[:,0]], axis=1))
            u = self.model(self.x)

            # Compute gradient u_x within the GradientTape
            # since we need second derivatives
            u_x = tape.gradient(u, self.x)

        u_xx = tape.gradient(u_x, self.x)

        del tape

        return self.fun_r(self.x, u, u_x, u_xx)
