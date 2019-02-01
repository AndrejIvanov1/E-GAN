"""Run a training job on Cloud ML Engine to train a GAN.
Usage:
  trainer.main --network-type <network-type> [--batch-size <batch-size>] [--disc-train-steps <disc-train-steps>]

Options:
  -h --help     Show this screen.
  --batch-size <batch-size>  Integer value indiciating batch size [default: 256]
  --disc-train-steps <disc-train-steps> Discriminator train steps [default: 2]
"""
from docopt import docopt
from trainer.generator import Generator
from trainer.discriminator import Discriminator
from trainer.generation import Generation
from trainer.egan import EGAN

import tensorflow as tf
import os

num_epochs = 6
noise_dim = 100
generator_batch_size = 16
discriminator_train_steps = 2
BUFFER_SIZE = 60000
BATCH_SIZE = 256
credentials_path = r'C:\Users\user\key.json'
network_type = 'EGAN'

def train(dataset, epochs):	
	gan = EGAN(num_parents=1, \
			   num_children=3, \
			   noise_dim=100, \
			   discriminator_update_steps=discriminator_train_steps)
	gan.train(dataset, epochs, batch_size=BATCH_SIZE)

def cloud_setup():
	checkpoints_path = os.path.join("checkpoints", "egan")
	if not os.path.exists(checkpoints_path):
		os.makedirs(checkpoints_path)


def local_setup():
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	if os.path.exists(credentials_path):
		os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

if __name__ == '__main__':
	arguments = docopt(__doc__)
	print("Arguments: ", arguments)

	network_type = arguments['<network-type>']
	discriminator_train_steps = int(arguments['--disc-train-steps'])
	BATCH_SIZE = int(arguments['--batch-size'])

	print(network_type, BATCH_SIZE, discriminator_train_steps)

	print("Running on {}".format(os.name))

	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5

	test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')
	test_images = (test_images - 127.5) / 127.5

	print(train_images.shape, train_labels.shape)
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE * discriminator_train_steps)

	local_setup()
	cloud_setup()
	train(train_dataset, num_epochs)

