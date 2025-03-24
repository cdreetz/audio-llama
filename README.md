# AudioLLM: Multimodal Language Model with Audio Understanding

AudioLLM is a research project that combines large language models (LLaMA) with audio processing capabilities (Whisper) to create a powerful multimodal system capable of understanding and responding to both text and audio inputs.

## Features

- Combines LLaMA language model with Whisper audio encoder
- Audio projection layer that maps Whisper features to LLaMA embedding space
- Parameter-efficient training with only projector and LoRA layers trainable
- Supports mixed precision training
- Comprehensive training, evaluation and inference pipelines

## Architecture

AudioLLM works by:

1. Processing audio inputs through the Whisper encoder
2. Projecting audio features into LLaMA's embedding space using a learnable projection layer
3. Integrating audio embeddings with text using special tokens (`<audio>`, `</audio>`)
4. Fine-tuning only the projection layer and LoRA adapters, keeping base models frozen

## Installation

```bash
# Clone the repository
git clone https://github.com/cdreetz/audio-llm.git
cd audio-llm

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

## Data

### LibriSpeech
```bash
cd src
./run_librispeech.sh
```

### Hugging Face (Recommended)
```bash
python download_huggingface.py --max-wer 5.0
# audio files saved in ./data/huggingface dir
# examples saved in instruction_examples.json

```

### Training

```bash
python train.py \
    --data_path /path/to/dataset.json \
    --audio_dir /path/to/audio/files \
    --llama_path meta-llama/Llama-3.2-3B-Instruct \
    --whisper_path openai/whisper-large-v3-turbo \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --num_epochs 5 \
    --learning_rate 5e-5
    --fp16
    --use_wandb
```

See `train.py` for additional training options.

### Inference

```bash
python inference.py \
# TODO
```

### Evaluation

```bash
# TODO
```

## Dataset Format

The dataset should be a JSON file with entries in the following format:

```json
[
  {
    "text": "Describe the audio: <audio>",
    "audio_paths": "sample1.wav",
    "response": "This is a recording of a piano playing."
  },
  {
    "text": "What can you hear in this recording? <audio>",
    "audio_paths": "sample2.wav",
    "response": "This recording contains a person speaking in English."
  }
]
```

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@misc{audioLLM2025,
  title = {AudioLLM: Multimodal Adapter for Audio Understanding},
  year = {2025},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

