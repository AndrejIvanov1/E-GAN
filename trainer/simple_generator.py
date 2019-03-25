import tensorflow as tf
import copy

class Generator:

	def __init__(self, model=None, noise_dim=100):
		self._noise_dim = noise_dim
		self._optimizer = tf.train.AdamOptimizer(1e-4)
		if not model:
			self._model = self._create_model()
		else:
			self._model = model


	def _create_model(self):
		"""
		model = tf.keras.Sequential()

		model.add(tf.keras.layers.Dense(128 * 7 * 7, activation="relu", input_dim=self._noise_dim))
		model.add(tf.keras.layers.Reshape((7, 7, 128)))

		model.add(tf.keras.layers.UpSampling2D())
		model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
		model.add(tf.keras.layers.Activation("relu"))

		model.add(tf.keras.layers.UpSampling2D())
		model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
		
		model.add(tf.keras.layers.Activation("relu"))
		
		model.add(tf.keras.layers.Conv2D(1, kernel_size=3, padding="same"))
		model.add(tf.keras.layers.Activation("tanh"))

		assert model.output_shape == (None, 28, 28, 1)"""
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(self._noise_dim,)))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		  
		model.add(tf.keras.layers.Reshape((7, 7, 256)))
		assert model.output_shape == (None, 7, 7, 256)
		
		model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
		assert model.output_shape == (None, 7, 7, 128) 
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
		assert model.output_shape == (None, 14, 14, 64)    

		model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
		assert model.output_shape == (None, 28, 28, 1)
	  
		return model

	def get_model(self):
		return self._model

	def set_model(self, new_model):
		self._model = new_model
		
	def get_weights(self):
		return self._model.get_weights()

	def set_weights(self, weights):
		self._model.set_weights(weights)

	def get_optimizer(self):
		return self._optimizer

	def variables(self):
		return self.get_model().variables	

	def set_mutation(self, mutation):
		self._mutation = mutation

	# Name of origin mutation(used only for debugging)
	def mutation(self):
		return self._mutation

	def clone(self, mutation='None'):
		model_clone = self._create_model()
		model_clone.set_weights(self.get_weights())

		new_generator = Generator(model=model_clone)

		new_generator.set_mutation(mutation)

		return new_generator

	def n_clones(self, mutations):
		return [self.clone(mutation=m.__name__) for m in mutations]

	def generate_images(self, noise, training=False):
		return self.get_model()(noise, training=training)

	def save_values(self):
		self._saved_values = self.values()

	def reset_values(self):
		self.assign_values(self._saved_values)

	def assign_values(self, new_values):
		for i in range(len(new_values)):
			val = new_values[i]
			self.variables()[i].assign(val)

	def values(self):
		return [var.value() for var in self.variables()]


	def loss(self, Dgz):
		return tf.losses.sigmoid_cross_entropy(tf.ones_like(Dgz), Dgz)