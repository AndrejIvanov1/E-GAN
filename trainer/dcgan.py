from  trainer.utils import generate_and_save_images, upload_file_to_cloud 
from  trainer.generator import Generator
from  trainer.discriminator import Discriminator

import time

class DCGAN:
	def __init__(self, noise_dim, discriminator_update_steps=2):
		self._discriminator = Discriminator()
		self._generator = Generator(noise_dim=noise_dim)

		self._noise_dim = noise_dim
		self._discriminator_update_steps = discriminator_update_steps

		self._num_examples_to_generate = 16
		self._random_vector_for_generation = tf.random_normal([self._num_examples_to_generate, noise_dim])

	def train(self, dataset, epochs, job_dir, batch_size=256):
		self._checkpoint_save_path = os.path.join(job_dir, "checkpoints", "egan")
		self._batch_size = batch_size

		noise_for_display_images = noise = tf.random_normal([self._num_examples_to_generate, self._noise_dim])

		for epoch in range(epochs):
			start_time = time.time()
			counter = 0
			for real_batch in dataset:
				self.train_step(real_batch)
				if counter % 10 == 0:
					pass
					#print("{} batched done".format(counter)) 
				if counter % 10 == 0:
					pass
					#print("Displaying images")
					#generate_and_save_images(self._generation.get_parent(), counter, noise_for_display_images)
				counter += 1
				break

			self.save_models()
			generate_and_save_images(self._generator, \
								     epoch, \
								     self._random_vector_for_generation, \
								     job_dir)
			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))


	def train_step(self, real_batch):
		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)
		start_time = time.time()
		for real_images in real_batch:
			self.disc_train_step(real_images) 

		print("Discriminator train step time:", time.time() - start_time)

		start_time = time.time()
		self.gen_train_step()
		print("Gen train step: ", time.time() - start_time)
		start_time = time.time()


	def disc_train_step(self, real_images):
		noise = tf.random_normal([self._batch_size, self._noise_dim])
		generated_images = self._generator.generate_images(noise)

		with tf.GradientTape() as disc_tape:
			real_output = self._discriminator.discriminate_images(real_images)
			generated_output = self._discriminator.discriminate_images(generated_images)

			if real_output.shape != generated_output.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(real_output.shape, generated_output.shape))
				return

			disc_loss = self._discriminator.loss(real_output, generated_output)
			print("Discriminator loss: ", disc_loss.numpy())

			gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
			self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))


	def save_models(self):
		discriminator_path = os.path.join(self._checkpoint_save_path, 'discriminator.h5')
		generator_path = os.path.join(self._checkpoint_save_path, 'generator.h5')
		tf.keras.models.save_model(self._discriminator.get_model(), discriminator_path)
		tf.keras.models.save_model(self._generation.get_parent().get_model(), generator_path)
		upload_file_to_cloud(discriminator_path)
		upload_file_to_cloud(generator_path)