gcloud ml-engine jobs submit training fashion_mnist_3_opts_lr_1_4_disc_lr_1_3 ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/fashion_mnist/3_opts_lr_1_4_disc_lr_1_3 ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type OLD_EGAN --dataset fashion --batch-size 256 --disc-train-steps 2 --epochs 6 --gamma 1.5