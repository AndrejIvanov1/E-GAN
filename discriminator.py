import tensorflow as tf
tf.enable_eager_execution()

class Discriminator:

	def __init__(self):
		self._model = self._create_model()
		self._optimizer = tf.train.AdamOptimizer(1e-4)

	def get_model(self):
		return self._model

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
		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	 
		return model

	def variables(self):
		return self.get_model().variables

	def discriminate_images(self, images):
		return self.get_model()(images, training=True)


	def loss(self, real_output, generated_output):
		real_loss = tf.losses.sigmoid_cross_entropy( \
			multi_class_labels=tf.ones_like(real_output), logits=real_output)

		#real_loss1 = tf.reduce_mean(tf.log(tf.reshape(real_output, (real_output.shape[0]))))
		#print(real_output.shape)
		#real_loss1 = tf.log(real_output)
		#real_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
		#real_loss2 = tf.reduce_mean(tf.log(real_output + 0.00001))
		#print(real_loss, real_loss1, real_loss2)

		generated_loss = tf.losses.sigmoid_cross_entropy( \
			multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

		total_loss = real_loss + generated_loss
		return total_loss