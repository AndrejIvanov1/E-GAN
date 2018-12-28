import tensorflow as tf
tf.enable_eager_execution()
from mutations import heuristic_mutation, minimax_mutation, least_square_mutation
import time
import os
import matplotlib.pyplot as plt
import fitness
import numpy as np
from utils import generate_and_save_images
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_examples_to_generate = 16
random_vector_for_generation = tf.random_normal([16, 100])

class EGAN:
	def __init__(self, discriminator, generation):
		self._discriminator = discriminator
		self._generation = generation
		self._discriminator_update_steps = 1
		self._gamma = 0.0

	def train(self, dataset, epochs, batch_size=256, noise_dim=100):
		#train_step = tf.contrib.eager.defun(self.train_step)
		self._batch_size = batch_size
		self._noise_dim = noise_dim

		noise_for_display_images = noise = tf.random_normal([16, self._noise_dim])
		for epoch in range(epochs):
			start_time = time.time()
			# self.generate_and_save_images(self._generation.get_parents()[0].get_model(), epoch, random_vector_for_generation)
			iterator = dataset.make_one_shot_iterator()
			self.train_step(iterator)
			generate_and_save_images(self._generation.get_parent(), epoch, noise_for_display_images)

			print ('Time taken for epoch {}: {} sec'.format(epoch + 1, time.time()-start_time))

	def train_step(self, dataset_iterator):
		real_images = None
		for step in range(self._discriminator_update_steps):
			real_images = dataset_iterator.get_next()
			self.disc_train_step(real_images)

		children = self.gen_train_step(mutations=[heuristic_mutation, minimax_mutation, least_square_mutation])

		self.selection(children, real_images)



	def disc_train_step(self, real_images):
		noise = tf.random_normal([self._batch_size, self._noise_dim])
		generated_images = self._generation.generate_images(noise)

		with tf.GradientTape() as disc_tape:
			real_output = self._discriminator.discriminate_images(real_images)
			generated_output = self._discriminator.discriminate_images(generated_images)

			assert real_output.shape == generated_output.shape == (self._batch_size, 1)

			disc_loss = self._discriminator.loss(real_output, generated_output)

			gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.variables())
			self._discriminator.get_optimizer().apply_gradients(zip(gradients_of_discriminator, self._discriminator.variables()))

	def gen_train_step(self, mutations):
		for parent in self._generation.get_parents():
			z = tf.random_normal([self._batch_size, self._noise_dim])

			# TODO: NEEDS TO BE IN PARALEL
			children = []
			for mutation in mutations:
				with tf.GradientTape() as gen_tape:
					child = parent.clone()
					Gz = child.generate_images(z, training=True)
					DGz = self._discriminator.discriminate_images(Gz)
					child_loss = mutation(DGz)
					print(mutation.__name__, child_loss)

					gradients_of_child = gen_tape.gradient(child_loss, child.variables())
					child.get_optimizer().apply_gradients(zip(gradients_of_child, child.variables()))

					children.append(child)

			return children

	def selection(self, children, real_images):
		Dx = self._discriminator.discriminate_images(real_images)
		z = tf.random_normal([self._batch_size, self._noise_dim])

		# TODO: MAKE IT PARALLEL
		fitnesses = []
		for child in children:
			DGz = self._calc_DGz(child, z)

			fitnesses.append(fitness.total_score(Dx, DGz, gamma=0.0))

		new_parents = fitness.select_fittest(fitnesses, children, n_parents=self._generation.get_num_parents())
		self._generation.new_generation(new_parents)

		# Works only with arrays of tensors ??
		#print(tf.map_fn(lambda child: fitness.total_score(Dx, self._calc_DGz(child, z)), np.array(fintesses)))

	def _calc_DGz(self, generator, z):
		Gz = generator.generate_images(z, training=True)
		return self._discriminator.discriminate_images(Gz)