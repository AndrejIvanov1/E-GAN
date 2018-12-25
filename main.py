from generator import Generator
from discriminator import Discriminator
from generation import Generation
import os

num_epochs = 1


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	generation = Generation(num_parents=1, num_children=3)
	generation.initialize()

	