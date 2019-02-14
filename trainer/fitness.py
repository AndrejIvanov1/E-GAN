import tensorflow as tf
from trainer.utils import flatten

# E[log(D(G))]
def quality_score(Dx, DGz):
	DGz = tf.nn.sigmoid(DGz)
	score = tf.reduce_mean(tf.log(DGz))
	#print("Quality: ", score.numpy())

	return score

def diversity_score(gradients):
	gradients = flatten(gradients)
	score = -tf.log(tf.norm(gradients))
	#print("Diversity: ", score.numpy())
	return score

def total_score(discriminator, x, Gz, gamma=0.15):

	with tf.GradientTape() as disc_tape:
		Dx = discriminator.discriminate_images(x)
		DGz = discriminator.discriminate_images(Gz)
		disc_loss = discriminator.loss(Dx, DGz)
		
	gradients = disc_tape.gradient(disc_loss, discriminator.variables())

	qs = quality_score(Dx, DGz)
	ds = diversity_score(gradients)

	score = (1.0 - gamma) * qs + (gamma * ds)
	#print("Total score: ", score.numpy())
	return score, qs, ds


def select_fittest(scored_children, n_parents=1):
	sorted_children = sorted(scored_children, key=lambda x: x[1], reverse=True)
	
	return zip(*sorted_children[:n_parents])
	#return list(map(lambda pair: pair[1], fitnesses[:n_parents]))