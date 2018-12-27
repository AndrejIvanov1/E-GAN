import tensorflow as tf
tf.enable_eager_execution()

class Generator:

	def __init__(self, model=None, noise_dim=100):
		self._noise_dim = noise_dim
		self._optimizer = tf.train.AdamOptimizer(1e-4)
		if not model:
			self._model = self._create_model()
		else:
			self._model = model


	def _create_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(self._noise_dim,)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())
		  
		model.add(tf.keras.layers.Reshape((7, 7, 256)))
		assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
		
		model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
		assert model.output_shape == (None, 7, 7, 128)  
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		assert model.output_shape == (None, 14, 14, 64)    
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
		assert model.output_shape == (None, 28, 28, 1)
	  
		return model

	def get_model(self):
		return self._model

	def get_weights(self):
		return self._model.get_weights()

	def get_optimizer(self):
		return self._optimizer

	def variables(self):
		return self.get_model().variables	

	def clone(self):
		model_clone = tf.keras.models.clone_model(self._model)
		model_clone.set_weights(self.get_weights())

		return Generator(model=model_clone)

	def generate_images(self, noise, training=False):
		return self.get_model()(noise, training=training)