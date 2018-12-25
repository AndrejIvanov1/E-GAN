from generator import Generator

class Generation:

	def __init__(self, num_parents=1, num_children=3):
		self._num_parents = num_parents
		self._num_children = num_children


	def initialize(self, noise_dim=100):
		self._parents = []
		for parent in range(self._num_parents):
			self._parents.append(Generator(noise_dim=100))

	def get_parents(self):
		return self._parents