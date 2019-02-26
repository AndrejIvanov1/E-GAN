from trainer.generator import Generator
import tensorflow as tf

class Generation:

	def __init__(self, num_parents=1, num_children=3):
		self._num_parents = num_parents
		self._num_children = num_children


	def initialize(self, noise_dim=100):
		self._parents = []
		for parent in range(self._num_parents):
			self._parents.append(Generator(noise_dim=100))

	def update_parent(self, index, new_weights):
		self._parents[index].set_weights(new_weights)

	def next_gen(self, new_weights):
		assert self._num_parents == len(new_weights)

		for i in range(self._num_parents):
			self.update_parent(i, new_weights[i])

	def new_generation(self, new_values):
		new_values = new_values[0]
		for i in range(len(new_values)):
			val = new_values[i]
			"""
			print(self.get_parent().variables()[i])
			print("VAL --------------", val) 
			print(self.get_parent().variables()[i].shape, val.shape)
			"""
			self.get_parent().variables()[i].assign(val)
		#self._parents = new_parents

	def get_parents(self):
		return self._parents

	def get_parent(self):
		return self.get_parents()[0]

	def get_num_parents(self):
		return self._num_parents

	def generate_images(self, noise):
		batch_size = noise.shape[0]
		parent_batch_size = int(batch_size) // self._num_parents
		curr_index = 0
		images_per_parent = []
		for parent in self.get_parents():
			images_batch = parent.generate_images(noise[curr_index:min(curr_index+parent_batch_size, batch_size),:], training=True)
			images_per_parent.append(images_batch)

		return tf.reshape(tf.stack(images_per_parent), (-1, 28, 28, 1))