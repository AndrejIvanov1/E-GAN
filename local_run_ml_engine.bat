gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type EGAN --dataset fashion --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/fashion_mnist/dcgan --epochs 25 --gamma 0.5