from pinn_num import * 

def get_data(L, h, u_in):
    # Generate boudary data (X_data, Y_data)
    ## Walls
    m_tmp = 100
    x_tmp = np.linspace(0, L, m_tmp)
    y_tmp = h*np.ones(m_tmp)
    X_data = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    x_tmp = np.linspace(0, L, m_tmp)
    y_tmp = np.zeros(m_tmp)
    X_tmp = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    X_data = np.concatenate((X_data,X_tmp),axis = 0)
    Y_data = np.zeros(X_data.shape)
    ## Inlet
    m_tmp = 40
    y_tmp = np.linspace(0, h, m_tmp+2)[1:-1]
    x_tmp = np.zeros(np.shape(y_tmp))
    X_tmp = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)
    Y_tmp = np.zeros(np.shape(X_tmp))
    Y_tmp[:,0] = u_in
    X_data = np.concatenate((X_data,X_tmp),axis = 0)
    Y_data = np.concatenate((Y_data,Y_tmp),axis = 0)
    # Generate Random point data (X_r)
    m_tmp1 = 100
    m_tmp2 = 30
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

    #X_r = np.expand_dims(X_r,2)
    #X_data = np.expand_dims(X_data,2)
    #Y_data = np.expand_dims(Y_data,2)

    X_r = X_r.astype(np.float32)
    X_data = X_data.astype(np.float32)
    Y_data = Y_data.astype(np.float32)
    

    return X_r, X_data, Y_data


def main():
    u_in = 0.034
    h = 0.05
    L = 20*h
    #Init data
    rho = 910
    mu = 0.3094
    nu = mu/rho
    print('hello')
    # Set Model 
    #model = NN_Model(3, layers= [50, 50, 50, 50, 50])
    model = get_model(2,3,layers = [30,30,30])
    # Get domain data and collocation points 
    colloc_point, X, Y = get_data(L, h, u_in)
    print(np.shape(colloc_point))
    print(np.shape(X))
    # Set PDE parameters 
    # Set solver 
    solver = StatNS_PINN(model,colloc_point, nu,pde_residual_scaling = 1e-1)
    solver.model.summary()
    # Set data points 
    print(solver.input_dim)
    print(np.shape(Y))
    print('hello')
    # Set optimizer 
    optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    # Solve 
    solver.solve_with_TFoptimizer(optim, X, Y, N=200)
    print('run solver model')
    solver.solve_with_ScipyOptimizer(X, Y,
                                     method='L-BFGS-B',
                                     options={'maxiter': 25000,
                                              'maxfun': 50000,
                                              'maxcor': 50,
                                              'maxls': 50,
                                              'ftol': 1.0*np.finfo(float).eps})
    # Plot resulting fields 
    m_tmp1 = 100
    m_tmp2 = 30
    x_tmp = np.linspace(0,L,m_tmp1+2)
    y_tmp = np.linspace(0,h,m_tmp2+2)
    x_tmp, y_tmp = np.meshgrid(x_tmp,y_tmp)
    x_tmp = x_tmp.flatten()
    y_tmp = y_tmp.flatten()
    X_disp = np.concatenate((np.expand_dims(x_tmp,1),np.expand_dims(y_tmp,1)),axis = 1)

    pred = solver.model(X_disp)
    u, v, p = pred[:,0], pred[:,1], pred[:,2]


    plt.scatter(x_tmp,y_tmp,c = p)
    #plt.tricontourf(x_tmp,y_tmp,u)
    #plt.tricontourf(xdata, ydata, zdata)
    plt.colorbar()
    plt.show()

    plt.scatter(x_tmp,y_tmp,c = u)
    #plt.tricontourf(x_tmp,y_tmp,u)
    #plt.tricontourf(xdata, ydata, zdata)
    plt.colorbar()
    plt.show()

    plt.scatter(x_tmp,y_tmp,c = v)
    #plt.tricontourf(x_tmp,y_tmp,u)
    #plt.tricontourf(xdata, ydata, zdata)
    plt.colorbar()
    plt.show()

if __name__ == '__main__' : 
    main()