from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow', 'numpy', 'h5py'] 

setup(
	name='trainer.main',
	version='0.1',
	author='Andrej Ivanov',
	author_email='ivandrej@gmail.com',
	install_requires=REQUIRED_PACKAGES,
	packages=find_packages(),
	include_package_data=True,
	description='A package for training an evolutionary generative adversarial network')