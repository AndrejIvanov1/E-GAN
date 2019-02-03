from  trainer.utils import generate_and_save_images, upload_file_to_cloud 
from  trainer.generator import Generator
from  trainer.discriminator import Discriminator

import tensorflow as tf
import os
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

			for real_batch in dataset:
				batch_start_time = time.time()
				self.train_step(real_batch)
				print("Time for batch: ", time.time() - batch_start_time)

			self.save_models()
			generate_and_save_images(self._generator, \
								     epoch, \
								     self._random_vector_for_generation, \
								     job_dir)
			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))

	def train_step(self, real_batch):
		#disc_train_step = tf.contrib.eager.defun(self.disc_train_step)
		#gen_train_step = tf.contrib.eager.defun(self.disc_train_step)

		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)

		with tf.GradientTape() as gen_tape:
			start_time = time.time()
			DGz = None
			for x in real_batch:
				Gz = self._calc_Gz()
				DGz = self._discriminator.discriminate_images(Gz)
				self.disc_train_step(x, DGz) 

			print("Discriminator train step time:", time.time() - start_time)

			start_time = time.time()
			Gz = self._calc_Gz()
			DGz = self._discriminator.discriminate_images(Gz)
			self.gen_train_step(DGz, gen_tape)
			print("Gen train step: ", time.time() - start_time)
			
	def _calc_Gz(self):
		z = tf.random_normal([self._batch_size, self._noise_dim])
		Gz = self._generator.generate_images(z)
		#DGz = self._discriminator.discriminate_images(Gz)

		return Gz


	def gen_train_step(self, DGz, gen_tape):
		gen_loss = self._generator.loss(DGz)

		gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.variables())
		self._generator.get_optimizer().apply_gradients(zip(gradients_of_generator, self._generator.variables()))


	def disc_train_step(self, x, DGz):
			
		with tf.GradientTape() as disc_tape:
			Dx = self._discriminator.discriminate_images(x)

			if Dx.shape != DGz.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(Dx.shape, DGz.shape))
				return

			disc_loss = self._discriminator.loss(Dx, DGz)
			# print("Discriminator loss: ", disc_loss.numpy())

		gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
		self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))


	def save_models(self):
		discriminator_path = os.path.join(self._checkpoint_save_path, 'discriminator.h5')
		generator_path = os.path.join(self._checkpoint_save_path, 'generator.h5')

		tf.keras.models.save_model(self._discriminator.get_model(), discriminator_path)
		tf.keras.models.save_model(self._generator.get_model(), generator_path)

		upload_file_to_cloud(discriminator_path)
		upload_file_to_cloud(generator_path)

