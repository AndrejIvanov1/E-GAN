from generator import Generator
import os

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	g = Generator()
	child = g.clone()

	print(child.get_model().layers == g.get_model().layers)