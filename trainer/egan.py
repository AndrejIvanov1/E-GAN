from   trainer.mutations import heuristic_mutation, minimax_mutation, least_square_mutation
import trainer.fitness as fitness
from   trainer.utils import generate_and_save_images, upload_file_to_cloud, upload_dir_to_cloud
from   trainer.generator import Generator
from   trainer.discriminator import Discriminator
from   trainer.generation import Generation
from   trainer.adam_optimizer import CustomAdamOptimizer

import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np

print("tf version: ", tf.__version__)
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class EGAN:
	def __init__(self, num_parents, num_children, noise_dim, discriminator_update_steps=2, gamma=0.4):
		self._generation = Generation(num_parents=1, num_children=3)
		self._generation.initialize(noise_dim=noise_dim)
		self._noise_dim = noise_dim

		self._discriminator = Discriminator()
		self._discriminator_update_steps = discriminator_update_steps
		self._gamma = gamma

		self._mutations = [heuristic_mutation, minimax_mutation, least_square_mutation]
		#self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
		self._optimizers = [tf.train.AdamOptimizer(learning_rate=1e-3) for mutation in self._mutations]

		self._num_examples_to_generate = 16
		self._random_vector_for_generation = tf.random_normal([self._num_examples_to_generate, noise_dim])

	def train(self, dataset, epochs, job_dir, batch_size=256, restore=False, n_iterations_loss_plot=2):
		self._checkpoint_save_path = os.path.join(job_dir, "checkpoints")
		self._batch_size = batch_size

		self._summary_path = os.path.join(job_dir, "summary")
		self._global_step = tf.train.get_or_create_global_step()
		self._summary_writer = tf.contrib.summary.create_file_writer(self._summary_path[18:], flush_millis=10000)
		self._summary_writer.set_as_default()
		tf.contrib.summary.always_record_summaries()


		noise_for_display_images = noise = tf.random_normal([self._num_examples_to_generate, self._noise_dim])
		for epoch in range(epochs):
			start_time = time.time()
			iteration = 0
			for real_batch in dataset:
				print("Iteration #{}".format(iteration))
				batch_time = time.time()
				record_loss = ((iteration % n_iterations_loss_plot) == 0)
				self.train_step(real_batch, record_loss=record_loss)

				self._global_step.assign_add(1)
				iteration+=1
								
			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))
			#self.save_models()
			generate_and_save_images(self._generation.get_parent(), \
								     epoch, \
								     self._random_vector_for_generation,
								     job_dir)
			upload_dir_to_cloud(self._summary_path[18:])

	#@tf.contrib.eager.defun
	def train_step(self, real_batch, record_loss=False):
		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)
		start_time = time.time()
		recorded = record_loss
		for real_images in real_batch:
			self.disc_train_step(real_images, recorded)
			recorded = True

		print("Discriminator train step time:", time.time() - start_time)

		start_time = time.time()
		children = self.gen_train_step(self._mutations,
							            real_batch[0], 
							            record_loss=record_loss)
		print("Gen train step: ", time.time() - start_time)

	def disc_train_step(self, x, record_loss):
		#global z, Gz, Dx, DGz, disc_loss, gradients_of_discriminator, disc_tape

		z = tf.random_normal([self._batch_size, self._noise_dim])
		Gz = self._generation.generate_images(z)

		with tf.GradientTape() as disc_tape:
			Dx = self._discriminator.discriminate_images(x, training=True)
			DGz = self._discriminator.discriminate_images(Gz, training=True)

			if Dx.shape != DGz.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(Dx.shape, DGz.shape))
				return

			disc_loss = self._discriminator.loss(Dx, DGz)
			if record_loss:
				with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
					tf.contrib.summary.scalar('Discriminator_loss', disc_loss)

		gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
		self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))

	def gen_train_step(self, mutations, x, record_loss):
		for parent in self._generation.get_parents():

			with tf.GradientTape(persistent=True) as tape:
				z = tf.random_normal([self._batch_size, self._noise_dim])
				losses = self.mutate_children(parent, mutations, tape, z, record_loss)

			z = tf.random_normal([self._batch_size, self._noise_dim])
			children_values, Gz = self.apply_gradients(tape, parent, losses, z)

			# Can defun
			self.selection(children_values, Gz, x)

	def mutate_children(self, parent, mutations, tape, z, record_loss):
		Gz = parent.generate_images(z, training=True)
		DGz = self._discriminator.discriminate_images(Gz, training=False) 

		return list(map(lambda mutation: self.mutate_child(parent, tape, mutation, DGz, record_loss), mutations))

	def apply_gradients(self, tape, parent, losses, z):
		parent.save_values()
		#z = tf.random_normal([self._batch_size, self._noise_dim])

		return zip(*list(map(lambda i: self.apply_gradient(tape, parent, losses[i], self._optimizers[0], z), range(len(losses)))))

	def apply_gradient(self, tape, parent, loss, optimizer, z):
		gradients= tape.gradient(loss, parent.variables())
		optimizer.apply_gradients(zip(gradients, parent.variables()))

		Gz = parent.generate_images(z, training=False)
		new_values = parent.values()
		parent.reset_values() 

		return new_values, Gz

	def mutate_child(self, parent, tape, mutation, DGz, record_loss):
		child_loss = mutation(DGz)
		
		if record_loss:
			with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				tf.contrib.summary.scalar(mutation.__name__, child_loss, family='mutations')

		return child_loss
			
			
	def selection(self, values, Gzs, x):
		fitnesses = list(map(lambda Gz: self.score_child(Gz, x), Gzs))

		scored_children = [(values[i], ) + fitnesses[i] for i in range(len(values))]
		new_values, fitnesses, quality, diversity = fitness.select_fittest(scored_children, n_parents=self._generation.get_num_parents())

		with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
			tf.contrib.summary.scalar('total_score', fitnesses[0], family='fitness')
			tf.contrib.summary.scalar('quality_score', quality[0], family='fitness')
			tf.contrib.summary.scalar('diversity_score', diversity[0], family='fitness')

		self._generation.new_generation(new_values)


	def score_child(self, Gz, x):
		return fitness.total_score(self._discriminator, x, Gz, gamma=self._gamma)

	"""
	def record_mutations(self, mutations, losses):
		assert len(mutations) == len(losses)

		for i in range(len(mutations)):
			with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				tf.contrib.summary.scalar(mutations[i].__name__, losses[i], family='mutations')

	def apply_gradients(self, parent, loss, tape, z):
		grad = tape.gradient(loss, parent.variables())
		# new optimizer every time
		optimizer = tf.train.AdamOptimizer(1e-4)
		optimizer.apply_gradients(zip(grad, parent.variables()))
		#parent.get_optimizer().apply_gradients(zip(grad, parent.variables()))

		new_weights = parent.get_weights()

		Gz = parent.generate_images(z)
		parent.set_weights(self._saved_weights)

		return new_weights, Gz

	def old_selection(self, scored_children):
		weights, fitnesses, quality, diversity = fitness.select_fittest(scored_children, n_parents=self._generation.get_num_parents())

		with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
			tf.contrib.summary.scalar('total_score', fitnesses[0], family='fitness')
			tf.contrib.summary.scalar('quality_score', quality[0], family='fitness')
			tf.contrib.summary.scalar('diversity_score', diversity[0], family='fitness')

		self._generation.next_gen(weights)

	def old_selection(self, children, real_images):
		z = tf.random_normal([self._batch_size, self._noise_dim])

		# TODO: MAKE IT PARALLEL
		fitnesses = []
		for child in children:
			Gz = child.generate_images(z, training=True)
			#fitnesses.append(fitness.total_score(self._discriminator, real_images, Gz, gamma=self._gamma))

		#print(fitnesses)
		new_parents = fitness.select_fittest(fitnesses, children, n_parents=self._generation.get_num_parents())
		#print("New generation: ", [parent.mutation() for parent in new_parents])
		self._generation.new_generation(new_parents)

		# Works only with arrays of tensors ??
		#print(tf.map_fn(lambda child: fitness.total_score(Dx, self._calc_DGz(child, z)), np.array(fintesses)))

	def _calc_DGz(self, generator, z):
		Gz = generator.generate_images(z, training=True)
		return self._discriminator.discriminate_images(Gz)


	def save_models(self):
		discriminator_path = os.path.join(self._checkpoint_save_path, 'discriminator.h5')
		generator_path = os.path.join(self._checkpoint_save_path, 'generator.h5')
		tf.keras.models.save_model(self._discriminator.get_model(), discriminator_path)
		tf.keras.models.save_model(self._generation.get_parent().get_model(), generator_path)
		upload_file_to_cloud(discriminator_path)
		upload_file_to_cloud(generator_path)
	"""