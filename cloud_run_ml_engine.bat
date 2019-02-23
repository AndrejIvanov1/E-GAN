gcloud ml-engine jobs submit training working_egan_disc_traning_false_memory_check ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/working_egan/disc_traning_false_memory_check ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --epochs 3 --gamma 0.5