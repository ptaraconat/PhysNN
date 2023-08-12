#from pinn_num import * 
import tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from optimizer import L_BFGS_B, TFOpt

def get_data(L, h, u_in):
    # Generate boudary data (X_data, Y_data)
    ## Walls
    m_tmp = 500
    x_tmp = np.linspace(0, L, m_tmp)
    y_tmp = h*np.ones(m_tmp)
    X_data = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    x_tmp = np.linspace(0, L, m_tmp)
    y_tmp = np.zeros(m_tmp)
    X_tmp = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    X_data = np.concatenate((X_data,X_tmp),axis = 0)
    Y_data = np.zeros(X_data.shape)
    ## Inlet
    m_tmp = 100
    y_tmp = np.linspace(0, h, m_tmp+2)[1:-1]
    x_tmp = np.zeros(np.shape(y_tmp))
    X_tmp = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    Y_tmp = np.zeros(np.shape(X_tmp))
    Y_tmp[:,0] = u_in
    X_data = np.concatenate((X_data,X_tmp),axis = 0)
    Y_data = np.concatenate((Y_data,Y_tmp),axis = 0)
    # Generate Random point data (X_r)
    m_tmp1 = 100
    m_tmp2 = 50
    x_tmp = np.linspace(0,L,m_tmp1+2)[1:-1]
    y_tmp = np.linspace(0,h,m_tmp2+2)[1:-1]
    x_tmp, y_tmp = np.meshgrid(x_tmp,y_tmp)
    x_tmp = x_tmp.flatten()
    y_tmp = y_tmp.flatten()
    X_r = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    # Plot data points
    plt.plot(X_r[:,0],X_r[:,1],'ro')
    plt.plot(X_data[:,0],X_data[:,1],'k*')
    plt.show()

    plt.scatter(X_r[:,0],X_r[:,1],c='black')
    plt.scatter(X_data[:,0],X_data[:,1],c = Y_data[:,0] )
    plt.colorbar()
    plt.show()

    plt.scatter(X_r[:,0],X_r[:,1],c='black')
    plt.scatter(X_data[:,0],X_data[:,1],c = Y_data[:,1] )
    plt.colorbar()
    plt.show()

    #X_r = np.expand_dims(X_r,2)
    #X_data = np.expand_dims(X_data,2)
    #Y_data = np.expand_dims(Y_data,2)

    X_r = X_r.astype(np.float32)
    X_data = X_data.astype(np.float32)
    Y_data = Y_data.astype(np.float32)
    

    return X_r, X_data, Y_data

def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

def contour(grid, x, y, z, title, levels=50):
    """
    Contour plot.

    Args:
        grid: plot position.
        x: x-array.
        y: y-array.
        z: z-array.
        title: title string.
        levels: number of contour lines.
    """

    # get the value range
    vmin = np.min(z)
    vmax = np.max(z)
    # plot a contour
    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.title(title)
    cbar = plt.colorbar(pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)

def main(maxiter = 2000, num_train_samples = 10000, num_test_samples = 100, maxiter_adam = 20000): 
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """

    #u0 = 0.034
    #h = 0.05
    #L = 20*h
    #Init data
    #rho = 910
    #mu = 0.3094
    #nu = mu/rho


    h = 1 
    L = 1 
    u0 = 1
    rho = 1
    nu = 0.01

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, rho=rho, nu=nu).build()

    # create training input
    xy_eqn = np.random.rand(num_train_samples, 2)
    xy_eqn[:,0] = xy_eqn[:,0] * L
    xy_eqn[:,1] = xy_eqn[:,1] * h
    xy_ub = np.random.rand(num_train_samples//2, 2)  # top-bottom boundaries
    xy_ub[..., 1] = np.round(xy_ub[..., 1])          # y-position is 0 or 1 
    xy_ub[:,0] = xy_ub[:,0] * L 
    xy_ub[:,1] = xy_ub[:,1] * h
    xy_lr = np.random.rand(num_train_samples//2, 2)  # left-right boundaries
    xy_lr[..., 0] = 0. #np.round(xy_lr[..., 0])          # x-position is 0 
    xy_lr[:,0] = xy_lr[:,0] * L
    xy_lr[:,1] = xy_lr[:,1] * h
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
    x_train = [xy_eqn, xy_bnd] 

    # create training output
    zeros = np.zeros((num_train_samples, 2))
    uv_bnd = np.zeros((num_train_samples, 2))
    index = np.where(xy_bnd[:,0] == 0.)[0]
    #uv_bnd[..., 0] = u0 * np.floor(xy_bnd[..., 1])
    uv_bnd[index,0] = u0
    y_train = [zeros, zeros, uv_bnd]

    plt.scatter(xy_eqn[:,0],xy_eqn[:,1],c = 'black')
    plt.scatter(xy_bnd[:,0],xy_bnd[:,1],c = uv_bnd[:,0])
    plt.colorbar()
    plt.show()

    plt.scatter(xy_eqn[:,0],xy_eqn[:,1],c = 'black')
    plt.scatter(xy_bnd[:,0],xy_bnd[:,1],c = uv_bnd[:,1])
    plt.colorbar()
    plt.show()

    # train the model with adam 
    optim = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    tfopt = TFOpt(model = pinn,x_train = x_train,y_train = y_train,optim = optim, maxiter = maxiter_adam)
    tfopt.fit()

    # train the model using L-BFGS-B algorithm
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train, maxiter=maxiter)
    lbfgs.fit()

    # create meshgrid coordinates (x, y) for test plots
    x = np.linspace(0, 1, num_test_samples)
    y = np.linspace(0, 1, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (psi, p)
    psi_p = network.predict(xy, batch_size=len(xy))
    psi, p = [ psi_p[..., i].reshape(x.shape) for i in range(psi_p.shape[-1]) ]
    # compute (u, v)
    u, v = uv(network, xy)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)

    # plot test results
    fig = plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 2)
    contour(gs[0, 0], x, y, psi, 'psi')
    contour(gs[0, 1], x, y, p, 'p')
    contour(gs[1, 0], x, y, u, 'u')
    contour(gs[1, 1], x, y, v, 'v')
    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(maxiter = 2000, num_train_samples = 10000, num_test_samples = 100)