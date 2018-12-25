from generator import Generator
from discriminator import Discriminator
from generation import Generation
import os

num_epochs = 1
discriminator_update_steps = 2
noise_dim = 100

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	generation = Generation(num_parents=1, num_children=3)
	generation.initialize(noise_dim=100)

	discriminator = Discriminator()

	for epoch in range(num_epochs):
		for step in range(discriminator_update_steps):
			# Sample batch of noise
			# Adam optimize discriminator
			pass
