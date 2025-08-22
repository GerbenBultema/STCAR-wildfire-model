from helper import generate_grid_adjacency, ClipConstraint
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class STCAR(tf.keras.Model):
    def __init__(self, use_bias=True, n_samples=100, distance=10, size=64):
        super().__init__()
        self.use_bias = use_bias
        self.n_samples = n_samples
        self.distance = distance
        self.size = size

        self.linear = tf.keras.layers.Dense(1, use_bias=self.use_bias)
        self.rho = self.add_weight(
            name="rho",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=ClipConstraint(0.0, 1.0)
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=(),
            initializer=tf.keras.initializers.Constant(1),
            trainable=False,
            constraint=ClipConstraint(0.0, 1.0)
        )
        self.r = self.add_weight(
            name="r",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=ClipConstraint(0.0, 1.0)
        )

    def call(self, X, training=False):

        self.W =  generate_grid_adjacency(X, self.distance, "exponential")
        self.W_D = tf.linalg.diag(tf.reduce_sum(self.W, axis=-1))
        self.precision = self.W_D - self.rho * self.W                                

        epsilon = 1e-5
        n_nodes = tf.shape(self.precision)[-1]
        self.cov = tf.linalg.inv(self.precision / self.sigma**2 + epsilon * tf.eye(n_nodes))

        mu = tf.squeeze(self.linear(X), axis=-1)                      
        mu = tf.reshape(mu, [-1]) 
        mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(self.cov))
        samples = mvn.sample(self.n_samples)                              

        Z1 = tf.reduce_mean(samples, axis=0)                          

        Z2 = self.r * tf.matmul(self.rho*tf.linalg.inv(self.W_D) @ self.W, Z1[..., tf.newaxis])     
        Z2 = tf.squeeze(Z2, axis=-1)                                    

        Z1 = tf.sigmoid(Z1)
        Z2 = tf.sigmoid(Z2)
        return {
        'Z1': Z1,
        'Z2': Z2
        }

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            Z1 = outputs['Z1']
            Z2 = outputs['Z2']
            loss_fn = tf.keras.losses.BinaryCrossentropy()

            y_true = tf.reshape(y_true, [tf.shape(y_true)[0]* tf.shape(y_true)[1]])
            y_prevmask = x[:,:,-1]
            y_prevmask = tf.reshape(y_prevmask, [tf.shape(y_prevmask)[0]* tf.shape(y_prevmask)[1]])

            sample_weights = tf.cast((y_prevmask >= 0) & (y_true >= 0), tf.float32) #ignore -1 values
            loss_Z1 = loss_fn(y_prevmask, Z1, sample_weight=sample_weights)
            loss_Z2 = loss_fn(y_true, Z2, sample_weight=sample_weights)
            loss = loss_Z1 + loss_Z2   

        # Apply gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {
            "loss": loss,
            "loss_Z1": loss_Z1,
            "loss_Z2": loss_Z2,
        }

class STCAR_nn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.linear = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None) 
        ])
        self.rho = self.add_weight(
            name="rho",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=ClipConstraint(0.0, 1.0)
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=(),
            initializer=tf.keras.initializers.Constant(1),
            trainable=False,
            constraint=ClipConstraint(0.0, 1.0)
        )
        self.r = self.add_weight(
            name="r",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=ClipConstraint(0.0, 1.0)
        )
        self.n_samples = 100
        self.distance = 10
        self.size = 64


    def call(self, X, training=False):

        self.W =  generate_grid_adjacency(X, self.distance, "exponential")
        self.W_D = tf.linalg.diag(tf.reduce_sum(self.W, axis=-1))
        self.precision = self.W_D - self.rho * self.W                               

        epsilon = 1e-5
        n_nodes = tf.shape(self.precision)[-1]
        self.cov = tf.linalg.inv(self.precision / self.sigma**2 + epsilon * tf.eye(n_nodes))

        mu = tf.squeeze(self.linear(X), axis=-1)                  
        mu = tf.reshape(mu, [-1])  
        mu = tf.clip_by_value(mu, -10, 10)
        mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(self.cov))
        samples = mvn.sample(self.n_samples)                               

        Z1 = tf.reduce_mean(samples, axis=0)                          

        Z2 = self.r * tf.matmul(self.rho*tf.linalg.inv(self.W_D) @ self.W, Z1[..., tf.newaxis])     
        Z2 = tf.squeeze(Z2, axis=-1)                                    

        Z1 = tf.sigmoid(Z1)
        Z2 = tf.sigmoid(Z2)
        return {
        'Z1': Z1,
        'Z2': Z2
        }

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            Z1 = outputs['Z1']
            Z2 = outputs['Z2']
            loss_fn = tf.keras.losses.BinaryCrossentropy()


            y_true = tf.reshape(y_true, [tf.shape(y_true)[0]* tf.shape(y_true)[1]])
            y_prevmask = x[:,:,-1]
            y_prevmask = tf.reshape(y_prevmask, [tf.shape(y_prevmask)[0]* tf.shape(y_prevmask)[1]])

            sample_weights = tf.cast((y_prevmask >= 0) & (y_true >= 0), tf.float32) # Ignore -1 values
            loss_Z1 = loss_fn(y_prevmask, Z1, sample_weight=sample_weights)
            loss_Z2 = loss_fn(y_true, Z2, sample_weight=sample_weights)
            loss = loss_Z1 + loss_Z2


        # Apply gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Return logs for metrics
        return {
            "loss": loss,
            "loss_Z1": loss_Z1,
            "loss_Z2": loss_Z2,
        }
