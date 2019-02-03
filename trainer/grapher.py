import matplotlib.pyplot as plt
import os

class Grapher():
	def __init__(self, name):
		self._values = []
		self._name = name

	def record(self, value):
		self._values.append(value)

	def plot(self, folder_path):
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		plt.plot(self._values)
		#plt.show()
		print("Plotting: ", self._values)