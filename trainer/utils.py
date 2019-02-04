import os
import matplotlib.pyplot as plt
import tensorflow as tf
from google.cloud import storage
from google.auth import compute_engine

BUCKET_NAME = "gan_datasets"
PROJECT_ID = "e-gan-225521"

def generate_and_save_images(generator, epoch, test_input, job_dir):
	dir_path = os.path.join(job_dir[18:], 'images')
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	predictions = generator.generate_images(test_input, training=False)

	fig = plt.figure(figsize=(4,4))
	  
	for i in range(predictions.shape[0]):
	    plt.subplot(4, 4, i+1)
	    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
	    plt.axis('off')
	    
	file_path =  job_dir[18:] + '/images/image_at_epoch_{:04d}.png'.format(epoch)   
	plt.savefig(file_path)
	upload_file_to_cloud(file_path)
	plt.clf()
	plt.cla()
	plt.close()

	#plt.show()

 
 # Flattens a list of tensors(do not need to be of the same shape)
def flatten(tensor_list):
	return tf.concat([tf.reshape(tensor, [-1]) for tensor in tensor_list], axis=0)

def download_from_cloud(source_blob_name):
	source_blob_name = source_blob_name.replace("\\", "/")
	# Should be extracted to its own method
	credentials = compute_engine.Credentials()
	storage_client = storage.Client(project=PROJECT_ID)
	bucket = storage_client.get_bucket(BUCKET_NAME)
	print("Soure blob name: ", source_blob_name)
	blob = bucket.get_blob(source_blob_name[18:])
	destination_file_name = source_blob_name[18:]
	blob.download_to_filename(destination_file_name)

	print('Blob {} downloaded to {}.'.format(
	    source_blob_name,
	    destination_file_name))

def download_dir_from_cloud(blob_dir):
	blob_dir = blob_dir.replace("\\", "/")
	print("Downloading dir: {}".format(blob_dir))
	bucket = connect_to_cloud_storage()
	blobs = bucket.list_blobs(prefix=blob_dir)

	for blob in blobs:
		blob.download_to_filename(blob.name)

def upload_dir_to_cloud(dir_path):
	print("Uploading dir {} to cloud".format(dir_path))
	for filename in os.listdir(dir_path):
		print("Filename", filename)
		upload_file_to_cloud(os.path.join(dir_path, filename))

def upload_file_to_cloud(source_file_name):
	destination_blob_name = source_file_name.replace("\\", "/")
	_upload_blob(BUCKET_NAME, source_file_name, destination_blob_name)


def _upload_blob(bucket_name, source_file_name, destination_blob_name):
	# Should be extracted to its own method
	credentials = compute_engine.Credentials()
	storage_client = storage.Client(project=PROJECT_ID)
	bucket = storage_client.get_bucket(bucket_name)
	blob = bucket.blob(destination_blob_name)

	blob.upload_from_filename(source_file_name)

	print('File {} uploaded to {}.'.format(
	    source_file_name,
	    destination_blob_name)) 

def connect_to_cloud_storage():
	credentials = compute_engine.Credentials()
	storage_client = storage.Client(project=PROJECT_ID)
	bucket = storage_client.get_bucket(BUCKET_NAME)

	return bucket