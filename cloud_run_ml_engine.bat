gcloud ml-engine jobs submit training fashion_mnist_egan_3_optimizers_1 ^
 	--module-name=trainer.main ^
 	--package-path=trainer ^
 	--job-dir=gs://gan_datasets/fashion_mnist/egan_3_optimizers ^
 	--staging-bucket=gs://gan_datasets ^
 	--python-version=3.5 ^
 	--runtime-version=1.12 ^
 	--scale-tier=BASIC_GPU ^
 	--region=europe-west1 ^
 	-- --network-type EGAN --dataset fashion --batch-size 256 --disc-train-steps 2 --epochs 9 --gamma 0.5