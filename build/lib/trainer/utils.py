import os
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_and_save_images(generator, epoch, test_input):
	if not os.path.exists('../images'):
		os.mkdir('../images')
	predictions = generator.generate_images(test_input, training=False)

	fig = plt.figure(figsize=(4,4))
	  
	for i in range(predictions.shape[0]):
	    plt.subplot(4, 4, i+1)
	    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
	    plt.axis('off')
	        
	plt.savefig('../images/image_at_epoch_{:04d}.png'.format(epoch))
	#plt.show()

 
 # Flattens a list of tensors(do not need to be of the same shape)
def flatten(tensor_list):
	return tf.concat([tf.reshape(tensor, [-1]) for tensor in tensor_list], axis=0)