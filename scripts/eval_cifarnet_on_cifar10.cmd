SET TRAIN_DIR=../tmp/cifarnet-model

SET DATASET_DIR=../tmp/cifar10


python ../eval_image_classifier.py ^
  --checkpoint_path=%TRAIN_DIR% ^
  --eval_dir=%TRAIN_DIR% ^
  --dataset_name=cifar10 ^
  --dataset_split_name=test ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=cifarnet
