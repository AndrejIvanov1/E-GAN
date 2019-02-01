gcloud ml-engine jobs submit training evolutionary_gan_train5 ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	--staging-bucket gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1

