from  trainer.utils import generate_and_save_images, upload_file_to_cloud, download_from_cloud
from  trainer.generator import Generator
from  trainer.discriminator import Discriminator
from trainer.grapher import Grapher

import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

		self._disc_loss_grapher = Grapher('discriminator_loss')
		self._gen_loss_grapher = Grapher('generator_loss')

	def train(self, dataset, epochs, job_dir, batch_size=256, n_iterations_loss_plot=8, restore=False):
		self._checkpoint_save_path = os.path.join(job_dir, "checkpoints", "dcgan")
		self._batch_size = batch_size

		if restore:
			pass 
			#self._restore_models(job_dir)

		noise_for_display_images = noise = tf.random_normal([self._num_examples_to_generate, self._noise_dim])

		for epoch in range(epochs):
			start_time = time.time()
			iteration = 0
			for real_batch in dataset:
				print("Iteration #{}".format(iteration))
				record_loss = False
				if iteration % n_iterations_loss_plot == 0:
					record_loss = True

				self.train_step(real_batch, record_loss=record_loss)
				iteration += 1
				break

			self.save_models()
			generate_and_save_images(self._generator, \
								     epoch, \
								     self._random_vector_for_generation, \
								     job_dir)
			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))
		
		self.plot_losses(os.path.join(job_dir[18:], "plots"))

	def train_step(self, real_batch, record_loss=False):
		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)
		start_time = time.time()
		for real_images in real_batch:
			self.disc_train_step(real_images, record_loss=record_loss) 

		#print("Discriminator train step time:", time.time() - start_time)

		start_time = time.time()
		self.gen_train_step(record_loss=record_loss)
		#print("Gen train step: ", time.time() - start_time)
		

	def gen_train_step(self, record_loss=False):
		with tf.GradientTape() as gen_tape:
			z = tf.random_normal([self._batch_size, self._noise_dim])
			Gz = self._generator.generate_images(z)
			DGz = self._discriminator.discriminate_images(Gz)
			gen_loss = self._generator.loss(DGz)

			if record_loss:
				pass
				#self._gen_loss_grapher.record(gen_loss.numpy())
			#print("Gen loss: ", gen_loss.numpy())

		gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.variables())
		self._generator.get_optimizer().apply_gradients(zip(gradients_of_generator, self._generator.variables()))
	

			
	def disc_train_step(self, real_images, record_loss=False):
		noise = tf.random_normal([self._batch_size, self._noise_dim])
		generated_images = self._generator.generate_images(noise)

		with tf.GradientTape() as disc_tape:
			real_output = self._discriminator.discriminate_images(real_images)
			generated_output = self._discriminator.discriminate_images(generated_images)

			if real_output.shape != generated_output.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(real_output.shape, generated_output.shape))
				return

			disc_loss = self._discriminator.loss(real_output, generated_output)
			#print("Discriminator loss: ", disc_loss.numpy())

			if record_loss:
				pass
				#self._disc_loss_grapher.record(disc_loss.numpy())

		gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
		self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))


	def save_models(self):
		discriminator_path = os.path.join(self._checkpoint_save_path, 'discriminator.h5')
		generator_path = os.path.join(self._checkpoint_save_path, 'generator.h5')

		tf.keras.models.save_model(self._discriminator.get_model(), discriminator_path[18:])
		tf.keras.models.save_model(self._generator.get_model(), generator_path[18:])

		upload_file_to_cloud(discriminator_path[18:])
		upload_file_to_cloud(generator_path[18:])

	def plot_losses(self, folder_path):
		self._disc_loss_grapher.plot(folder_path)
		self._gen_loss_grapher.plot(folder_path)


	def _restore_models(self, job_dir):
		local_checkpoint_path = self._checkpoint_save_path[18:]
		discriminator_path = os.path.join(self._checkpoint_save_path, 'discriminator.h5')
		generator_path = os.path.join(self._checkpoint_save_path, 'generator.h5')

		if not os.listdir(local_checkpoint_path):
			#os.makedirs(local_checkpoint_path)

			download_from_cloud(discriminator_path)
			download_from_cloud(generator_path)

		restored_generator_model = tf.keras.models.load_model(generator_path[18:])
		restored_discriminator_model = tf.keras.models.load_model(discriminator_path[18:])

		self._generator.set_model(restored_generator_model)
		self._discriminator.set_model(restored_discriminator_model)

		# Need to load epoch too
