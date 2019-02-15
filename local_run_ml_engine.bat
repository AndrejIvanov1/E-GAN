gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/new_score_combination/one_disc_step --epochs 15 --gamma 0.7