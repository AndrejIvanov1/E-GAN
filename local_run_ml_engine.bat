gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/working_egan/new_optimizer_technique --epochs 10 --gamma 0.5