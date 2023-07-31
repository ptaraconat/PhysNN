import tensorflow as tf 

class PINN_Model(tf.keras.Model): 

    def __init__(self,input_dim,output_dim,layers = [20, 20, 20],**kwargs):
        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (input_dim,1))
        self.output_layer = tf.keras.layers.Dense(output_dim)
        self.hidden_layers = [tf.keras.layers.Dense(n_units, activation = 'tanh') for n_units in layers]
    
    def call(self,input):
        x = np.input_layer(x)
        for layer in self.hidden_layers : 
            x = layer(x)
        output = self.output_layer(x)
        return output

if __name__ == '__main__':
    print('hello')
    model = PINN_Model(2,3)
    model.build()
    model.summary()
    
