gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type OLD_EGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/old_gan\gamma_0_25_ --epochs 5 --gamma 0.25