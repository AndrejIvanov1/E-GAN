gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type OLD_EGAN --dataset fashion --batch-size 256 --disc-train-steps 1 --job-dir gs://gan_datasets/fashion_mnist/one_disc_step --epochs 10 --gamma 0.5