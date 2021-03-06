"""Run a training job on Cloud ML Engine to train a GAN.
Usage:
  trainer.main --network-type <network-type> --dataset <dataset-path> [--batch-size <batch-size>] [--disc-train-steps <disc-train-steps>] [--epochs <epochs>] [--restore] [--gamma <gamma>] [--job-dir <job-dir>]

Options:
  -h --help     Show this screen.
  --network-type <network-type> Type of GAN: EGAN or DCGAN
  --dataset <dataset-path> Path to dataset on GCloud
  --batch-size <batch-size>  Integer value indiciating batch size [default: 256]
  --disc-train-steps <disc-train-steps> Discriminator train steps [default: 2]
  --job-dir <job-dir> Job dir [default: '.']
  --restore
  --epochs <epochs> [default: 10]
  --gammma <gamma> [default: 0.4]
"""
from docopt import docopt
from trainer.generator import Generator
from trainer.discriminator import Discriminator
from trainer.generation import Generation
from trainer.egan import EGAN
from trainer.dcgan import DCGAN
from trainer.old_egan import OLD_EGAN
from trainer.utils import clean_dir, show_random_image

import tensorflow as tf
import os

num_epochs = 4
noise_dim = 100
generator_batch_size = 16
discriminator_train_steps = 2
BUFFER_SIZE = 60000
BATCH_SIZE = 256
credentials_path = r'C:\Users\user\key.json'
network_type = 'EGAN'
dataset_path = ''
JOB_DIR = '.'
restore = False
gamma = 0.4

def train(dataset, epochs):
	if network_type == 'EGAN':
		gan = EGAN(num_parents=1, \
				   num_children=3, \
				   noise_dim=100, \
				   discriminator_update_steps=discriminator_train_steps,
				   gamma=gamma)
	elif network_type == 'OLD_EGAN':
		gan = OLD_EGAN(num_parents=1, \
					   num_children=3, \
					   noise_dim=100, \
					   discriminator_update_steps=discriminator_train_steps,
					   gamma=gamma)
	else:
		gan = DCGAN(noise_dim=noise_dim, discriminator_update_steps=discriminator_train_steps)

	gan.train(dataset, epochs, job_dir=JOB_DIR, batch_size=BATCH_SIZE, restore=restore)

def cloud_setup():
	discriminator_checkpoints_path = os.path.join(JOB_DIR[18:], "checkpoints", "discriminator")
	generator_checkpoints_path = os.path.join(JOB_DIR[18:], "checkpoints", "generator")
	summary_path = os.path.join(JOB_DIR[18:], "summary")

	print(summary_path)
	if not os.path.exists(discriminator_checkpoints_path):
		os.makedirs(discriminator_checkpoints_path)
	if not os.path.exists(generator_checkpoints_path):
		os.makedirs(generator_checkpoints_path)
	if not os.path.exists(summary_path):
		os.makedirs(summary_path)

	clean_dir(summary_path)


def local_setup():
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	if os.path.exists(credentials_path):
		os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

if __name__ == '__main__':
	arguments = docopt(__doc__)
	print("Arguments: ", arguments)

	JOB_DIR = arguments['--job-dir']
	network_type = arguments['--network-type']
	dataset = arguments['--dataset']
	discriminator_train_steps = int(arguments['--disc-train-steps'])
	BATCH_SIZE = int(arguments['--batch-size'])
	num_epochs = int(arguments['--epochs'])
	restore = arguments['--restore']
	gamma = float(arguments['<gamma>'])

	if dataset == 'fashion':
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
	else:
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5

	test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')
	test_images = (test_images - 127.5) / 127.5


	print(train_images.shape, train_labels.shape)
	#show_random_image(train_images)
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE * discriminator_train_steps)

	local_setup()
	cloud_setup()
	train(train_dataset, num_epochs)

