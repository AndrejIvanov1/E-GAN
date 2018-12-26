import tensorflow as tf
tf.enable_eager_execution()


class EGAN:
	def __init__(self, discriminator, generation):
		self._discriminator = discriminator
		self._generation = generation
		self._discriminator_update_steps = 1

	def train(self, dataset, epochs, batch_size=256, noise_dim=100):
		self._batch_size = batch_size
		self._noise_dim = noise_dim
		for epoch in range(epochs):
			iterator = dataset.make_one_shot_iterator()
			self.train_step(iterator)

	def train_step(self, dataset_iterator):
		for step in range(self._discriminator_update_steps):
			real_images = dataset_iterator.get_next()
			noise = tf.random_normal([self._batch_size, self._noise_dim])
			generated_images = self._generation.generate_images(noise)

			with tf.GradientTape() as disc_tape:
				real_output = self._discriminator.discriminate_images(real_images)
				generated_output = self._discriminator.discriminate_images(generated_images)

				assert real_output.shape == generated_output.shape == (self._batch_size, 1)

				disc_loss = self._discriminator.loss(real_output, generated_output)

				#disc_gradient
				
