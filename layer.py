import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, xy):
        """
        Computing derivatives for the steady Navier-Stokes equation.

        Args:
            xy: input variable.

        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """

        x, y = [ xy[..., i, tf.newaxis] for i in range(xy.shape[-1]) ]
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(x)
            ggg.watch(y)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(x)
                gg.watch(y)
                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)
                    psi_p = self.model(tf.concat([x, y], axis=-1))
                    psi = psi_p[..., 0, tf.newaxis]
                    p   = psi_p[..., 1, tf.newaxis]
                u   =  g.batch_jacobian(psi, y)[..., 0]
                v   = -g.batch_jacobian(psi, x)[..., 0]
                p_x =  g.batch_jacobian(p,   x)[..., 0]
                p_y =  g.batch_jacobian(p,   y)[..., 0]
                del g
            u_x = gg.batch_jacobian(u, x)[..., 0]
            u_y = gg.batch_jacobian(u, y)[..., 0]
            v_x = gg.batch_jacobian(v, x)[..., 0]
            v_y = gg.batch_jacobian(v, y)[..., 0]
            del gg
        u_xx = ggg.batch_jacobian(u_x, x)[..., 0]
        u_yy = ggg.batch_jacobian(u_y, y)[..., 0]
        v_xx = ggg.batch_jacobian(v_x, x)[..., 0]
        v_yy = ggg.batch_jacobian(v_y, y)[..., 0]
        del ggg

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy

        return psi, p_grads, u_grads, v_grads

class GradientLayer_HarmOsc(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the harmonic oscilattor equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, time):
        """
        Computing derivatives for the steady Navier-Stokes equation.

        Args:
            time: input variable.

        Returns:
            u: prediction
            u_x: prediction first derivatives
            u_xx: prediction second derivatives
        """

        with tf.GradientTape(persistent=True) as tape:

            # Variable x is watched during tape
            # to compute derivatives u_x and u_xx
            tape.watch(time)

            # Determine residual
            #u = model(tf.stack([t[:,0], x[:,0]], axis=1))
            u = self.model(time)

            # Compute gradient u_x within the GradientTape
            # since we need second derivatives
            u_x = tape.gradient(u, time)

        u_xx = tape.gradient(u_x, time)

        del tape

        return u, u_x, u_xx

