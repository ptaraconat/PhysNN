from pinn_num import * 

def main():
    print('hello')
    # Set Model 
    #model = NN_Model(1)
    model = get_model(1,1,layers = [33, 33, 33])
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
    #solver.solve_with_TFoptimizer(optim, x_data, y_data, N=1000)
    solver.solve_with_ScipyOptimizer(x_data, y_data,
                                     method='L-BFGS-B',
                                     options={'maxiter': 25000,
                                              'maxfun': 50000,
                                              'maxcor': 50,
                                              'maxls': 50,
                                              'ftol': 1.0*np.finfo(float).eps})
    print(1.0*np.finfo(float).eps)
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
    
    
