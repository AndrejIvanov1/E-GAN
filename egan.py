import tensorflow as tf
tf.enable_eager_execution()


class EGAN:
	def __init__(self, discriminator, generation):
		self._discriminator = discriminator
		self._generation = generation
		self._discriminator_update_steps = 1

	def train(self, dataset, epochs):
		for epoch in range(epochs):
			iterator = dataset.make_one_shot_iterator()
			self.train_step(iterator)

	def train_step(self, dataset_iterator):
		print("Here")
		for step in range(self._discriminator_update_steps):
			real_images = dataset_iterator.get_next()
			noise = tf.random_normal([BATCH_SIZE, noise_dim])

			with tf.GradientTape() as disc_tape:
				generated_images = generation.generate_images(noise)
				
				print(real_images.shape)
				print(generated_images.shape)
