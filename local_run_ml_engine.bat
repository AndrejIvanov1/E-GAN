gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type OLD_EGAN --dataset fashion --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/fashion_mnist/old_egan --epochs 10 --gamma 0.5