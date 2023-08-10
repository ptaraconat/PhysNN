from pinn import * 
import numpy as np 
import matplotlib.pyplot as plt 
from optimizer import * 

def get_model(input_dim,output_dim,layers): 
    input = tf.keras.Input(shape = (input_dim,))
    x = input
    for n_units in layers : 
        x = tf.keras.layers.Dense(n_units, activation = 'tanh',kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(output_dim,activation = 'linear')(x)
    return tf.keras.Model(inputs = input, outputs = x)



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

def main():

    mass = 1
    viscosity = 4
    stiffness = 400
    N_training_data = 1000
    maxiter = 2000

    # build a core network model
    network = get_model(1,1,layers = [33, 33, 33])
    network.summary()
    # build a PINN model
    pinn = PINN_HarmOsc(network, mass, viscosity, stiffness).build()

    # create training input
    x_colloc = np.linspace(0,1,N_training_data)
    y_colloc = np.zeros(N_training_data)
    x_data = np.linspace(0,0.4,N_training_data)
    d_param = viscosity/(2*mass)
    w0_param = np.sqrt(stiffness/mass)
    y_data = oscillator(d_param, w0_param, x_data)

    x_data = np.expand_dims(x_data,1).astype(np.float32)
    y_data = np.expand_dims(y_data,1).astype(np.float32)
    x_colloc = np.expand_dims(x_colloc,1).astype(np.float32)
    y_colloc = np.expand_dims(y_colloc,1).astype(np.float32)

    plt.figure()
    plt.plot(x_colloc, oscillator(d_param,w0_param,x_colloc), label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()

    # train the model using L-BFGS-B algorithm
    x_train = [x_data,x_colloc]
    y_train = [y_data,y_colloc]

    yhat = network(x_colloc)
    plt.plot(x_colloc,yhat,'ro')
    plt.plot(x_colloc,oscillator(d_param,w0_param,x_colloc),'k-')
    plt.show() 

    optim = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    tfopt = TFOpt(model = pinn,x_train = x_train,y_train = y_train,optim = optim, maxiter = 20000)
    tfopt.fit()

    plt.plot(tfopt.hist)
    plt.show()
    plt.close()
    # Plot Result
    yhat = network(x_colloc)
    plt.plot(x_colloc,yhat,'ro')
    plt.plot(x_colloc,oscillator(d_param,w0_param,x_colloc),'k-')
    plt.show() 


    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, maxiter=maxiter)
    lbfgs.fit()

    plt.plot(lbfgs.hist)
    plt.show()
    plt.close()
    # Plot Result
    yhat = network(x_colloc)
    plt.plot(x_colloc,yhat,'ro')
    plt.plot(x_colloc,oscillator(d_param,w0_param,x_colloc),'k-')
    plt.show() 

    return pinn, network, lbfgs

    


if __name__ == '__main__':
    pinn, network, lbfgs = main()
    
    
