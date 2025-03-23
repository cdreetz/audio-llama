python train.py \
	--data_path /home/ubuntu/audio-llama/src/librispeech_data/audio_instruction_examples.json \
	--audio_dir /home/ubuntu/audio-llama/src/librispeech_data/audio/ \
	--llama_path meta-llama/Llama-3.2-3B-Instruct \
	--whisper_path openai/whisper-large-v3-turbo \
	--output_dir ./checkpoints \
	--fp16 \
	--use_wandb 
