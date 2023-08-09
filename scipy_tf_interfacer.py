import tensorflow as tf 
import scipy.optimize
import numpy as np 
import matplotlib.pyplot as plt 

# In order to run scipy optimizer on tensorflow model, on should provide the model 
# parameters as a 1D array. 
def stack_tf_model_weights(model):
    '''
    Stack the model parameters into a 1D array 
    Input : 
    model ::: tf.keras.Model ::: MLP model 
    Outputs : 
    flatten_weights ::: 
    shapes :::
    '''
    shapes = []
    # loop over model weights/biases 
    # loop from the input layer toward the output layer 
    # The weight matrice comes first, then the biases array comes
    for w in model.variables:
        shapes.append(w.shape)

    flatten_weights = np.concatenate([ w.flatten() for w in model.get_weights() ])
    return flatten_weights, shapes

def set_model_weights(model, flatten_weights, shapes): 
    # Check shapes : 
    if len(model.variables) != len(shapes):
        print('variables length missmatch')
        return None 
    else : 
        for w, shape in zip(model.variables,shapes):
            if w.shape != shape : 
                print('layers shape missmatch')
                return None 
    # loop over model varaibles 
    idx = 0 
    for w in model.variables : 
        shape = w.shape 
        if len(shape) == 2 : #if weight matrice 
            no_weights = shape[0]*shape[1]
        if len(shape) == 1 : # if biases array 
            no_weights = shape[0]
        
        flatten_weights_portion = flatten_weights[idx:idx+no_weights]
        
        idx += no_weights
        reshaped_weights_portion = flatten_weights_portion.reshape(shape)
        w.assign(reshaped_weights_portion)

class ModelInterfacer():
    def __init__(self, model, loss_func, x_data, y_data): 
        self.loss = loss_func
        self.inputs = x_data
        self.outputs = y_data 
        self.model = model 
        _, shapes = stack_tf_model_weights(self.model)
        self.variables_shapes = shapes
        self.hist = []
        self.iter = 0 
        self.current_loss = 0.
    
    @tf.function
    def get_grad(self):

        with tf.GradientTape(persistent = True) as tape :
            tape.watch(self.model.trainable_variables)
            y_hat = self.model(self.inputs)
            loss = tf.reduce_mean(self.loss(y_hat,self.outputs))
        gradients = tape.gradient(loss,self.model.trainable_variables) 
        del tape 

        return loss, gradients
    
    def get_weights(self) : 
        return stack_tf_model_weights(self.model)
    
    def __call__(self,flatten_weights):
        set_model_weights(self.model, flatten_weights, self.variables_shapes)
        loss, gradients = self.get_grad()
        loss = loss.numpy().astype('float64')
        self.current_loss = loss 
        grads = np.concatenate([ g.numpy().flatten() for g in gradients ]).astype('float64')
        return loss, grads
    
    def callback(self, xr=None):
        if self.iter % 1 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.iter+=1
 
################################
# Define some data 
x = np.linspace(0,10,5000)
y = np.sqrt(x)
# Define and train model
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape = (1,), activation = 'relu'),
                             tf.keras.layers.Dense(10,activation = 'relu'),
                             tf.keras.layers.Dense(1,activation = 'linear')])
model.compile(optimizer = 'adam', loss = 'MSE')
model.summary()
model.fit(x,y,epochs = 10,verbose = 1)
# Get model weights as a flatten array 
flat_w, shapes_w = stack_tf_model_weights(model)
##### Test set_model_weigths function #####
print('==============================================')
print('====== Test set_model_weights function =======')
model2 = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape = (1,), activation = 'relu'),
                             tf.keras.layers.Dense(10,activation = 'relu'),
                             tf.keras.layers.Dense(1,activation = 'linear')])

print('Before setting model2 weights with flatten_weights')
print('model1(3) prediction :', model(np.array([3])))
print('model2(3) prediction :',model2(np.array([3])))
print('model1(3) == model2(3) => ',model(np.array([3])) == model2(np.array([3])))
set_model_weights(model2, flat_w, shapes_w)
print('After setting model2 weights with flatten_weights')
print('model1(3) prediction :',model(np.array([3])))
print('model2(3) prediction :',model2(np.array([3])))
print('model1(3) == model2(3) => ',model(np.array([3])) == model2(np.array([3])))

########### Train with scipy optimizer ##########
model3 = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape = (1,), activation = 'relu'),
                             tf.keras.layers.Dense(10,activation = 'relu'),
                             tf.keras.layers.Dense(1,activation = 'linear')])

interfacer = ModelInterfacer(model, tf.keras.losses.mean_squared_error, x, y)
init_guess = interfacer.get_weights()[0]
#scipy.optimize.fmin_l_bfgs_b(func=interfacer, x0= init_guess,factr=10, pgtol=1e-10, m=50,maxls=50, maxiter=200,callback = interfacer.callback())
#scipy.optimize.fmin_l_bfgs_b(func=interfacer, x0= init_guess,callback = interfacer.callback())

#scipy.optimize.minimize(fun=interfacer,x0=init_guess,jac=True,callback=interfacer.callback)

results = scipy.optimize.minimize(fun = interfacer, 
                                  x0 = init_guess, 
                                  method = 'L-BFGS-B',
                                  jac = True, 
                                  callback = interfacer.callback,
                                  options = {'iprint' : 0,
                                             'maxiter': 50000,
                                             'maxfun' : 50000,
                                             'maxcor' : 500,
                                             'maxls': 500,
                                             'gtol': 0,
                                             'ftol' : 0})

plt.plot(x,interfacer.model(x),'ro')
plt.plot(x,y,'k-')
plt.show()

################
from optimizer import * 

x = np.linspace(0,10,100)
y = np.sqrt(x)

lbfgs = L_BFGS_B(model3, [x], [y])
lbfgs.fit()

plt.plot(x,lbfgs.model(x),'ro')
plt.plot(x,y,'k-')
plt.show()