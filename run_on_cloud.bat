gcloud ml-engine local train ^
	--module-name E-GAN.main.py ^
	--job-dir %cd%\Output ^
	--package-path %cd%
