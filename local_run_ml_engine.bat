gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --job-dir local --network-type DCGAN --batch-size 256 --disc-train-steps 2