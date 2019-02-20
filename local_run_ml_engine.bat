gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/disc_no_train/test --epochs 1 --gamma 0.4