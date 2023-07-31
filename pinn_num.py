import tensorflow as tf 

class PINN_Model(tf.keras.Model): 

    def __init__(self,input_dim,output_dim,layers = [20, 20, 20]):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (input_dim,1))
        self.output_layer = tf.keras.layers.Dense(output_dim)
        self.hidden_layers = [tf.keras.layers.Dense(n_units, activation = 'tanh') for n_units in layers]

if __name__ == '__main__':
    print('hello')
    
