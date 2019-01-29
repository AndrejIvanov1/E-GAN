import tensorflow as tf
tf.enable_eager_execution()
from generator import Generator
from discriminator import Discriminator
from generation import Generation
from egan import EGAN
import os

num_epochs = 6
noise_dim = 100
generator_batch_size = 16
discriminator_train_steps = 2
BUFFER_SIZE = 60000
BATCH_SIZE = 256


def train(dataset, epochs):
	generation = Generation(num_parents=1, num_children=3)
	generation.initialize(noise_dim=100)

	discriminator = Discriminator()
	
	gan = EGAN(discriminator, generation, discriminator_update_steps=discriminator_train_steps)
	gan.train(dataset, epochs, batch_size=BATCH_SIZE, noise_dim=100)

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5

	test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')
	test_images = (test_images - 127.5) / 127.5

	print(train_images.shape, train_labels.shape)
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE * discriminator_train_steps)

	train(train_dataset, num_epochs)

