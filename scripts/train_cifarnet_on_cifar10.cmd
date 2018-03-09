SET TRAIN_DIR=../tmp/cifarnet-model

SET DATASET_DIR=../tmp/cifar10


python ../download_and_convert_data.py --dataset_name=cifar10 --dataset_dir=%DATASET_DIR%


python ../train_image_classifier.py ^
  --train_dir=%TRAIN_DIR% ^
  --dataset_name=cifar10 ^
  --dataset_split_name=train ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=cifarnet ^
  --preprocessing_name=cifarnet ^
  --max_number_of_steps=100000 ^
  --batch_size=128 ^
  --save_interval_secs=120 ^
  --save_summaries_secs=120 ^
  --log_every_n_steps=100 ^
  --optimizer=sgd ^
  --learning_rate=0.1 ^
  --learning_rate_decay_factor=0.1 ^
  --num_epochs_per_decay=200 ^
  --weight_decay=0.004
