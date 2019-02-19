import tensorflow as tf

class Discriminator:

	def __init__(self):
		self._model = self._create_model()
		self._optimizer = tf.train.AdamOptimizer(1e-4)

	def get_model(self):
		return self._model

	def set_model(self, new_model):
		self._model = new_model

	def get_optimizer(self):
		return self._optimizer
	
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

	def variables(self):
		return self.get_model().variables

	def discriminate_images(self, images, training=True):
		return self.get_model()(images, training=training)


	# -E[log(D(x))] - E[log(1 - D(G(z)))]
	def loss(self, real_output, generated_output):
		real_loss = tf.losses.sigmoid_cross_entropy( \
			multi_class_labels=tf.ones_like(real_output), logits=real_output)

		generated_loss = tf.losses.sigmoid_cross_entropy( \
			multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

		total_loss = real_loss + generated_loss
		return total_loss