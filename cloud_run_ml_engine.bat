gcloud ml-engine jobs submit training new_score_combination_gamma_0_2  ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/new_score_combination_folder/gamma_0_2 ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --epochs 15 --gamma 0.2