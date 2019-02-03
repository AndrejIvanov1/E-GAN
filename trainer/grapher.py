from trainer.utils import upload_file_to_cloud
import matplotlib.pyplot as plt
import os

class Grapher():
	def __init__(self, name):
		self._values = []
		self._name = name

	def record(self, value):
		self._values.append(value)

	def plot(self, folder_path):
		print("Plotting: ", self._name)
		plt.close() 
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)

		plt.plot(self._values)
		file_path = os.path.join(folder_path, self._name)
		print("File path: ", file_path)
		plt.savefig(file_path)

		upload_file_to_cloud(file_path + ".png")