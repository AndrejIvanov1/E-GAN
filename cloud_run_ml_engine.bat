gcloud ml-engine jobs submit training old_egan_gamma_0_4_more_epochs  ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/old_egan/gamma_0_4_more_epochs ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type OLD_EGAN --batch-size 256 --disc-train-steps 2 --epochs 12 --gamma 0.4