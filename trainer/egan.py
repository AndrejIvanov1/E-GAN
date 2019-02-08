from   trainer.mutations import heuristic_mutation, minimax_mutation, least_square_mutation
import trainer.fitness as fitness
from   trainer.utils import generate_and_save_images, upload_file_to_cloud 
from   trainer.generator import Generator
from   trainer.discriminator import Discriminator
from   trainer.generation import Generation

import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np

print("tf version: ", tf.__version__)
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class EGAN:
	def __init__(self, num_parents, num_children, noise_dim, discriminator_update_steps=2, gamma=0.2):
		self._generation = Generation(num_parents=1, num_children=3)
		self._generation.initialize(noise_dim=noise_dim)
		self._noise_dim = noise_dim

		self._discriminator = Discriminator()
		self._discriminator_update_steps = discriminator_update_steps
		self._gamma = gamma

		self._num_examples_to_generate = 16
		self._random_vector_for_generation = tf.random_normal([self._num_examples_to_generate, noise_dim])

	def train(self, dataset, epochs, job_dir, batch_size=256, restore=False, n_iterations_loss_plot=1):
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

				if iteration == 1:
					break
				self._global_step.assign_add(1)
				iteration+=1
				
			#self.save_models()
			generate_and_save_images(self._generation.get_parent(), \
								     epoch, \
								     self._random_vector_for_generation,
								     job_dir)
			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))

	def train_step(self, real_batch, record_loss=False):
		real_batch = tf.split(real_batch, self._discriminator_update_steps, axis=0)
		start_time = time.time()
		recorded = record_loss
		for real_images in real_batch:
			self.disc_train_step(real_images, recorded)
			recorded = True

		print("Discriminator train step time:", time.time() - start_time)

		start_time = time.time()
		children = self.gen_train_step([heuristic_mutation, 
							            minimax_mutation, 
							            least_square_mutation],
							            real_batch[0], 
							            record_loss=record_loss)
		print("Gen train step: ", time.time() - start_time)
		start_time = time.time()
		#self.selection(children, real_batch[0])
		print("Selection: ", time.time() - start_time)


	def disc_train_step(self, real_images, record_loss):
		noise = tf.random_normal([self._batch_size, self._noise_dim])
		generated_images = self._generation.generate_images(noise)

		with tf.GradientTape() as disc_tape:
			real_output = self._discriminator.discriminate_images(real_images)
			generated_output = self._discriminator.discriminate_images(generated_images)

			if real_output.shape != generated_output.shape:
				print("D real output shape: {} does not match D generated output shape: {}".format(real_output.shape, generated_output.shape))
				return

			disc_loss = self._discriminator.loss(real_output, generated_output)
			if record_loss:
				with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
					tf.contrib.summary.scalar('Discriminator_loss', disc_loss)

		gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
		self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))

	def gen_train_step(self, mutations, x, record_loss):
		print("NUM PARENTS", self._generation.get_parents())
		for parent in self._generation.get_parents():
			print("GEN TRAIN STEP ------------------")
			self._saved_weights = parent.get_weights()
		
			with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
				tape.watch(parent.variables())
				z = tf.random_normal([self._batch_size, self._noise_dim])
				Gz = parent.generate_images(z)
				DGz = self._discriminator.discriminate_images(Gz)

				children_losses = list(map(lambda mutation: mutation(DGz), mutations))
			
			children_weights = list(map(lambda loss: self.apply_gradients(parent, loss, tape), children_losses))
			
				#children = list(map(lambda mutation: self.mutate(parent, mutation, z, DGz, x, tape, record_loss), mutations))
			#children = [self.mutate(z, parent, mutation, record_loss) for mutation in mutations]
			
			"""
			children = []
			for mutation in mutations:
				child = self.mutate(z, parent, mutation, record_loss)
				children.append(child)
			
			return children
	
			"""

	def apply_gradients(self, parent, loss, tape):
		grad = tape.gradient(loss, parent.variables())
		parent.get_optimizer().apply_gradients(zip(grad, parent.variables()))

		new_weights = parent.get_weights()
		parent.set_weights(self._saved_weights)

		return new_weights

	def mutate(self, parent, mutation, z, DGz, x, tape, record_loss):
		print("VARIABLES", parent.variables()[0])
		print("WEIGHTS", parent.get_weights()[0])

		return (1,1)
		# Create child by mutating parent
		child_loss = mutation(DGz)
		if record_loss:
			with self._summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				tf.contrib.summary.scalar(mutation.__name__, child_loss)
		grad = tape.gradient(child_loss, parent.variables())
		parent.get_optimizer().apply_gradients(zip(grad, parent.variables()))

		# Calculate fitness
		Gz = parent.generate_images(z)
		score = fitness.total_score(self._discriminator, x, Gz, gamma=self._gamma)

		result = (parent.get_weights(), score)

		# Move method to Generator object
		parent.set_weights(self._saved_weights)

		return result


	def selection(self, children, real_images):
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