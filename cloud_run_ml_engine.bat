gcloud ml-engine jobs submit training dcgan_50_epochs_GPU ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/dcgan_50_epochs ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type DCGAN --batch-size 256 --disc-train-steps 2 --epochs 10