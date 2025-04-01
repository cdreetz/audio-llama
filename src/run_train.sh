python train.py \
	--data_path  data/librispeech_full/examples.json \
	--audio_dir audio/ \
	--dataset_config data/librispeech_full/dataset_config.json \
	--batch_size 8 \
	--fp16 \
	--num_workers 16 \
	--use_wandb 
