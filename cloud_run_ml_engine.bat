gcloud ml-engine jobs submit training fashion_mnist_new_archs_old_egan_gamma_2 ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/fashion_mnist/new_archs_egan_gamma_2 ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type OLD_EGAN --dataset fashion --batch-size 256 --disc-train-steps 2 --epochs 6 --gamma 2