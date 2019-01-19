import tensorflow as tf
tf.enable_eager_execution()

def quality_score(Dx, DGz):
	DGz = tf.nn.sigmoid(DGz)
	score = tf.reduce_mean(tf.log(DGz))

	return score

def diversity_score(Dx, DGz):
	return 1

def total_score(Dx, DGz, gamma=0.0):
	return (1 - gamma) * quality_score(Dx, DGz) + \
			gamma      * diversity_score(Dx, DGz)


def select_fittest(fitnesses, children, n_parents=1):
	fitnesses = sorted(zip(fitnesses, children), key=lambda x: x[0], reverse=True)

	return list(map(lambda pair: pair[1], fitnesses[:n_parents]))