import tensorflow as tf
tf.enable_eager_execution

# L(G) = -E[log(D(G(z)))]
def heuristic_mutation(Dgz):
	#loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(Dgz), Dgz)
	Dgz = tf.nn.sigmoid(Dgz)
	loss = -tf.reduce_mean(tf.log(Dgz))
 
	return loss

# L(G) = E[log(1- D(G(z)))]
def minimax_mutation(Dgz):
	Dgz = tf.nn.sigmoid(Dgz)
	loss = tf.reduce_mean(tf.log(1.0 - Dgz))

	return loss


# L(G) = E[(D(G(z)) - 1)^2]
def least_square_mutation(Dgz):

	Dgz = tf.nn.sigmoid(Dgz)
	# loss = tf.reduce_mean((Dgz - 1.0) * (Dgz - 1.0))
	loss1 = tf.losses.mean_squared_error(tf.ones_like(Dgz), Dgz)

	print(loss, loss1)
	return loss