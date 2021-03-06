gcloud ml-engine jobs submit training simple_models_2_mutations ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/simple_models/2_mutations ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC ^
 	--region=europe-west1 ^
 	-- --network-type OLD_EGAN --dataset mnist --batch-size 256 --disc-train-steps 2 --epochs 10 --gamma 1.5