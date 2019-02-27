gcloud ml-engine jobs submit training variable_implementation_1_optimizer_separate_z ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/variable_implementation/1_optimizer_separate_z ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --epochs 10 --gamma 0.3