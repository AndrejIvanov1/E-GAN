gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type EGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/variable_implementation/pers_tape_same_z --epochs 10 --gamma 0.5