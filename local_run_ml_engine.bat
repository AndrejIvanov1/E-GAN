gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type OLD_EGAN --dataset mnist --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/simple_models/gamma_1_5 --epochs 10 --gamma 1.5