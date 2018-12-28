import tensorflow as tf
tf.enable_eager_execution()

def quality_score(Dx, DGz):
	print("Calculating quality score with GGz.shape={}".format(DGz.shape))

	print("Quality score", DGz)
	return 0

def diversity_score(Dx, DGz):
	return 0

def total_score(Dx, DGz, gamma=0.2):
	return (1 - gamma) * quality_score(Dx, DGz) + \
			gamma      * diversity_score(Dx, DGz)


def select_fittest(fitnesses, children, n_parents=1):
	fitnesses = sorted(zip(fitnesses, children), key=lambda x: x[0], reverse=True)

	return list(map(lambda pair: pair[1], fitnesses[:n_parents]))