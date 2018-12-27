import tensorflow as tf
tf.enable_eager_execution

def heuristic_mutation(Dgz):
	return tf.losses.sigmoid_cross_entropy(tf.ones_like(Dgz), Dgz)