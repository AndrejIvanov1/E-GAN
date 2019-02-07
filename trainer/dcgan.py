from  trainer.utils import generate_and_save_images, upload_file_to_cloud, download_from_cloud, upload_dir_to_cloud, download_dir_from_cloud
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

	def train(self, dataset, epochs, job_dir, batch_size=256, n_iterations_loss_plot=3, restore=False):
		self._discriminator_checkpoint_path = os.path.join(job_dir, "checkpoints", "discriminator")
		self._generator_checkpoint_path = os.path.join(job_dir, "checkpoints", "generator")
		self._discriminator_checkpoint = tfe.Checkpoint(optimizer=self._discriminator.get_optimizer(), model=self._discriminator.get_model())
		self._generator_checkpoint = tfe.Checkpoint(optimizer=self._generator.get_optimizer(), model=self._generator.get_model())
		self._summary_path = os.path.join(job_dir, "summary")
		self._global_step = tf.train.get_or_create_global_step()
		summary_writer = tf.contrib.summary.create_file_writer(self._summary_path[18:], flush_millis=10000)
		summary_writer.set_as_default()
		tf.contrib.summary.always_record_summaries()

		self._batch_size = batch_size

		if restore:
			self._restore_models(job_dir)

		noise_for_display_images = noise = tf.random_normal([self._num_examples_to_generate, self._noise_dim])

		with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
			for epoch in range(epochs):
				start_time = time.time()
				iteration = 0
				for real_batch in dataset:
					print("Iteration #{}".format(iteration))
					batch_time = time.time()
					record_loss = False
					if iteration % n_iterations_loss_plot == 0:
						record_loss = True

					self.train_step(real_batch, record_loss=record_loss)
					iteration += 1
					print("Batch time: ", time.time() - batch_time)

					self._global_step.assign_add(1)

				if epoch > 0 and epoch % 9 == 0:
					self.save_models()

				if epoch % 3 == 0:
					generate_and_save_images(self._generator, \
										     epoch, \
										     self._random_vector_for_generation, \
										     job_dir)
				print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))
	
		upload_dir_to_cloud(self._summary_path[18:])

	@tf.contrib.eager.defun
	def train_step(self, real_batch, record_loss=False):
		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)
		start_time = time.time()
		recorded = record_loss
		DGz = None
		with tf.GradientTape() as gen_tape:
			for x in real_batch:
				with tf.GradientTape() as disc_tape:
					z = tf.random_normal([self._batch_size, self._noise_dim])
					Gz = self._generator.generate_images(z)
					DGz = self._discriminator.discriminate_images(Gz)
					Dx = self._discriminator.discriminate_images(x)

					if Dx.shape != DGz.shape:
						print("D real output shape: {} does not match D generated output shape: {}".format(Dx.shape, DGz.shape))
						return

					disc_loss = self._discriminator.loss(Dx, DGz)

					if recorded:
						tf.contrib.summary.scalar('Discriminator_loss', disc_loss)

				gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
				self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))
				recorded = False 

			start_time = time.time()
			z = tf.random_normal([self._batch_size, self._noise_dim])
			Gz = self._generator.generate_images(z)
			DGz = self._discriminator.discriminate_images(Gz)
			gen_loss = self._generator.loss(DGz)
			if record_loss:
				tf.contrib.summary.scalar('Generator_loss', gen_loss)

		gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.variables())
		self._generator.get_optimizer().apply_gradients(zip(gradients_of_generator, self._generator.variables()))
		
	def gen_train_step(self, record_loss=False):
		with tf.GradientTape() as gen_tape:
			z = tf.random_normal([self._batch_size, self._noise_dim])
			Gz = self._generator.generate_images(z)
			DGz = self._discriminator.discriminate_images(Gz)
			gen_loss = self._generator.loss(DGz)

			#print("Global step: ", tf.train.get_global_step())
			if record_loss:
				tf.contrib.summary.scalar('Generator_loss', gen_loss)
				#self._gen_loss_grapher.record(gen_loss.numpy())
			#print("Gen loss: ", gen_loss.numpy())

		gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.variables())
		self._generator.get_optimizer().apply_gradients(zip(gradients_of_generator, self._generator.variables()))

	def disc_train_step(self, x, DGz, disc_tape, record_loss=False):
		with tf.GradientTape() as disc_tape:
			Dx = self._discriminator.discriminate_images(x)

			if Dx.shape != DGz.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(Dx.shape, DGz.shape))
				return

			disc_loss = self._discriminator.loss(Dx, DGz)

			if record_loss:
				tf.contrib.summary.scalar('Discriminator_loss', disc_loss)

		gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
		self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))


	def save_models(self):
		discriminator_prefix = os.path.join(self._discriminator_checkpoint_path, "checkpoint")
		generator_prefix = os.path.join(self._generator_checkpoint_path, "checkpoint")

		self._discriminator_checkpoint.save(file_prefix=discriminator_prefix[18:])
		self._generator_checkpoint.save(file_prefix=generator_prefix[18:])

		upload_dir_to_cloud(self._discriminator_checkpoint_path[18:])
		upload_dir_to_cloud(self._generator_checkpoint_path[18:])

	def plot_losses(self, folder_path):
		self._disc_loss_grapher.plot(folder_path)
		self._gen_loss_grapher.plot(folder_path)


	def _restore_models(self, job_dir):
		local_discriminator_path = self._discriminator_checkpoint_path[18:]
		local_generator_path = self._generator_checkpoint_path[18:]

		if not os.listdir(local_discriminator_path):
			download_dir_from_cloud(self._discriminator_checkpoint_path[18:])
			download_dir_from_cloud(self._generator_checkpoint_path[18:])

		status = self._discriminator_checkpoint.restore(tf.train.latest_checkpoint(self._discriminator_checkpoint_path[18:]))
		status = self._generator_checkpoint.restore(tf.train.latest_checkpoint(self._generator_checkpoint_path[18:]))

		# Need to load epoch too
