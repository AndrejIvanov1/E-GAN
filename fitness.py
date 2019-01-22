import tensorflow as tf
tf.enable_eager_execution()
from utils import flatten

# E[log(D(G))]
def quality_score(Dx, DGz):
	DGz = tf.nn.sigmoid(DGz)
	score = tf.reduce_mean(tf.log(DGz))

	return score

def diversity_score(gradients):
	gradients = flatten(gradients)
	print(gradients.shape)
	return 1

def total_score(discriminator, x, Gz, gamma=0.0):

	with tf.GradientTape() as disc_tape:
		Dx = discriminator.discriminate_images(x)
		DGz = discriminator.discriminate_images(Gz)
		disc_loss = discriminator.loss(Dx, DGz)
		
	gradients = disc_tape.gradient(disc_loss, discriminator.variables())

	score = (1 - gamma) * quality_score(Dx, DGz) + \
			gamma       * diversity_score(gradients)
	#print("Fitness score: ", score.numpy())

	return score


def select_fittest(fitnesses, children, n_parents=1):
	fitnesses = sorted(zip(fitnesses, children), key=lambda x: x[0], reverse=True)

	return list(map(lambda pair: pair[1], fitnesses[:n_parents]))