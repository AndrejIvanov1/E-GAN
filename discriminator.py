import tensorflow as tf
tf.enable_eager_execution()

class Discriminator:

	def __init__(self):
		self._model = self._create_model()

	
	def _create_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.3))
		  
		model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.3))
		   
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(1))
	 
		return model