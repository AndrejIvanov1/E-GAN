gcloud ml-engine local train ^
 	--module-name=trainer.main ^
 	--package-path=trainer\ ^
 	-- --network-type DCGAN --batch-size 256 --disc-train-steps 2 --job-dir gs://gan_datasets/checkpoint_test --epochs 1 --restore