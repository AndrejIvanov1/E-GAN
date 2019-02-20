gcloud ml-engine jobs submit training fixed_gan_correct  ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/fixed_gan/gamma_0_4 ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --epochs 6 --gamma 0.4