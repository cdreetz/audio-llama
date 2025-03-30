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

# Comparing Phi-4-Multimodal and Qwen2-Audio Approaches to Audio Processing

## Architecture Comparison

| Feature                | Phi-4-Multimodal                                                      | Qwen2-Audio                          |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------ |
| **Base Architecture**  | Mixture-of-LoRAs over frozen LLM                                      | Full fine-tuning approach            |
| **Total Parameters**   | 5.6B (3.8B LLM + 0.46B audio encoder + 0.46B LoRA + other components) | 8.2B total                           |
| **Base LLM**           | Phi-4-Mini (3.8B)                                                     | Qwen-7B                              |
| **Audio Encoder**      | Conformer-based (460M parameters)                                     | Whisper-large-v3 based               |
| **Integration Method** | Low-rank adaptation (LoRA)                                            | Direct fine-tuning of complete model |
| **Context Length**     | 128K tokens (~2.8 hours of audio)                                     | 128K tokens                          |

## Audio Processing Approach

| Aspect                   | Phi-4-Multimodal                                                 | Qwen2-Audio                                                |
| ------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| **Audio Tokenization**   | 80ms per token (750 tokens for 1 minute)                         | 40ms per token (1500 tokens for 1 minute)                  |
| **Modality Integration** | Modality-specific LoRAs with routing                             | Natural language prompts                                   |
| **Training Strategy**    | Two-stage: pre-training + instruction tuning with frozen encoder | Three-stage: pre-training, SFT, and DPO                    |
| **Multimodal Support**   | Full integration with vision and speech via separate LoRAs       | Two interaction modes (audio analysis and voice chat)      |
| **Preservation of LLM**  | Maintains original LLM capabilities (frozen base)                | Modified LLM weights that may impact original capabilities |

## Performance Comparison

| Benchmark                 | Phi-4-Multimodal                     | Qwen2-Audio                           |
| ------------------------- | ------------------------------------ | ------------------------------------- |
| **ASR (OpenASR)**         | 6.14 WER (ranks #1)                  | 7.43 WER                              |
| **ASR (CommonVoice)**     | 6.80 WER                             | 8.55 WER                              |
| **Speech Translation**    | Better on CoVoST2                    | Competitive on certain language pairs |
| **Audio Understanding**   | 6.98/10 on AIR-Bench                 | 6.93/10 on AIR-Bench                  |
| **Speech Summarization**  | Strong capability (up to 30 min)     | Limited to 30-second clips            |
| **Model Size Efficiency** | Better performance with smaller size | Larger model size                     |

## Key Architectural Differences

1. **Integration Philosophy**:

- **Phi-4-Multimodal**: Uses LoRA adapters to preserve the base language model capabilities while adding audio processing
- **Qwen2-Audio**: Replaces hierarchical tags with natural language prompts, directly fine-tuning the entire model

2. **Modality Handling**:

- **Phi-4-Multimodal**: Explicit routing between modalities through separate LoRAs
- **Qwen2-Audio**: Automatic detection of modality through prompt structure

3. **Audio Processing**:

- **Phi-4-Multimodal**: Conformer-based architecture with higher compression ratio
- **Qwen2-Audio**: Whisper-based architecture with finer-grained resolution

4. **Deployment Advantages**:

- **Phi-4-Multimodal**: More parameter-efficient, better for resource-constrained environments
- **Qwen2-Audio**: More integrated approach, potentially better for end-to-end applications

Both models represent significant advances in multimodal AI, but Phi-4-Multimodal's mixture-of-LoRAs approach offers better parameter efficiency and modality isolation, while Qwen2-Audio's direct fine-tuning may provide more seamless integration between modalities.
