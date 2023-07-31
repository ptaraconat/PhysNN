import tensorflow as tf 
import numpy as np 
import scipy.optimize
import matplotlib.pyplot as plt 


class NN_Model(tf.keras.Model): 

    def __init__(self,output_dim,layers = [10, 10, 10],**kwargs):
        super().__init__(**kwargs)
        self.output_layer = tf.keras.layers.Dense(output_dim)
        self.hidden_layers = [tf.keras.layers.Dense(n_units, activation = 'tanh') for n_units in layers]
    
    def call(self,input):
        x = input
        for layer in self.hidden_layers : 
            x = layer(x)
        return self.output_layer(x)

class PINNSolver():

    def __init__(self, model, X_r):
        self.model = model
        self.input_dim = np.shape(X_r)[1]
        self.model.build(input_shape=(None,self.input_dim))
        self.collocation = tf.convert_to_tensor(X_r)
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
    
    def loss_fn(self, X, u):
        
        # Compute phi_r
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r

        # Initialize loss
        u_pred = self.model(X)
        loss = phi_r + tf.reduce_mean(tf.keras.losses.mean_squared_error(u, u_pred))

        # Add phi_0 and phi_b to the loss
        #for i in range(len(X)):
        #    u_pred = self.model(X[i])
        #    loss += tf.reduce_mean(tf.square(u[i] - u_pred))
        
        return loss
    
    def get_grad(self, X, u):
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

def main():
    print('hello')
    # Set Model 
    model = NN_Model(1)
    # Set collocation points 
    X_r = np.linspace(0,1,500)
    X_r = np.expand_dims(X_r,1).astype(np.float32)
    # Set PDE parameters 
    mass = 1
    viscosity = 4
    stiffness = 400
    # Set solver 
    solver = HarmOsci_PINN(model,X_r, mass, viscosity, stiffness)
    solver.model.summary()
    # Set data points 
    def oscillator(d, w0, x):
        """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
        assert d < w0
        w = np.sqrt(w0**2-d**2)
        phi = np.arctan(-d/w)
        A = 1/(2*np.cos(phi))
        cos = np.cos(phi+w*x)
        sin = np.sin(phi+w*x)
        exp = np.exp(-d*x)
        y  = exp*2*A*cos
        return y
    X_r = np.linspace(0,1,500)
    x_data = X_r[0:200:20]
    d_param = viscosity/(2*mass)
    w0_param = np.sqrt(stiffness/mass)
    y_data = oscillator(d_param, w0_param, x_data)
    x_data = np.expand_dims(x_data,1).astype(np.float32)
    y_data = np.expand_dims(y_data,1).astype(np.float32)
    #
    plt.figure()
    plt.plot(X_r, oscillator(d_param,w0_param,X_r), label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()
    # Set optimizer 
    optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    # Solve 
    solver.solve_with_TFoptimizer(optim, x_data, y_data, N=25000)
    solver.plot_loss_history(ax=None)
    plt.show()
    plt.close()
    # Plot Result
    yhat = solver.model(solver.x)
    plt.plot(solver.x,yhat,'ro')
    plt.plot(solver.x,oscillator(d_param,w0_param,solver.x),'k-')
    plt.show() 

    


if __name__ == '__main__':
    main()
    
    
