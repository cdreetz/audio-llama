import os
import torch
import logging
import argparse
import wandb
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import LlamaTokenizer, WhisperProcessor, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from models.allm import AudioLLM
from dataloaders import create_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train AudioLLM model")
    
    # Model paths
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path to pre-trained LLaMA model")
    parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v3-turbo",
                        help="Path to pre-trained Whisper model")
    
    # Data paths
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON file with dataset entries")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, 
                        help="Evaluation batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, 
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--lora_rank", type=int, default=32, 
                        help="Rank for LoRA adapter")
    
    # Other parameters
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="Evaluate every X steps")
    parser.add_argument("--log_steps", type=int, default=100, 
                        help="Log every X steps")
    parser.add_argument("--max_audio_length", type=int, default=30, 
                        help="Maximum audio length in seconds")
    parser.add_argument("--text_max_length", type=int, default=512, 
                        help="Maximum text length in tokens")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="audio-llm", 
                        help="W&B project name")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_checkpoint(model, optimizer, scheduler, step, epoch, args, final=False):
    """Save model checkpoint"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if final:
        checkpoint_path = os.path.join(args.output_dir, "final_checkpoint")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Get trainable parameters state dict
    trainable_params_dict = {
        'projector': model.projector.state_dict(),
        'lora_layers': {name: layer.state_dict() for name, layer in model.lora_layers.items()}
    }
    
    checkpoint = {
        'model': trainable_params_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'step': step,
        'epoch': epoch,
        'args': args
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint.pt"))
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def evaluate(model, val_dataloader, device, fp16=False):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_features = batch.get("audio_features")
            if audio_features is not None:
                audio_features = audio_features.to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_features=audio_features,
                    labels=labels
                )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def train(args):
    """Main training function"""
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logging
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args)
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizers/processors
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    whisper_processor = WhisperProcessor.from_pretrained(args.whisper_path)
    
    # Add special tokens for audio
    audio_tokens = {"additional_special_tokens": ["<audio>", "</audio>"]}
    llama_tokenizer.add_special_tokens(audio_tokens)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        data_path=args.data_path,
        audio_dir=args.audio_dir,
        whisper_model_id=args.whisper_path,
        llama_model_id=args.llama_path,
        batch_size=args.batch_size,
        max_audio_length=args.max_audio_length,
        text_max_length=args.text_max_length,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Initialize model
    logger.info("Initializing AudioLLM model...")
    model = AudioLLM(
        llama_path=args.llama_path,
        whisper_path=args.whisper_path,
        lora_rank=args.lora_rank
    )
    model.llama.model.resize_token_embeddings(len(llama_tokenizer))
    
    # Set tokenizer in model
    model.tokenizer = llama_tokenizer
    
    # Move model to device
    model.to(device)
    
    # Get trainable parameters
    trainable_params = model.get_trainable_params()
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Initialize optimizer
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.grad_accum_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Initialize amp scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_features = batch.get("audio_features")
            if audio_features is not None:
                audio_features = audio_features.to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_features=audio_features,
                    labels=labels
                )
                
                loss = outputs.loss / args.grad_accum_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation steps reached
            if (step + 1) % args.grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # Increment global step
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item() * args.grad_accum_steps})
                
                # Log metrics
                if global_step % args.log_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    
                    # Log to tensorboard
                    writer.add_scalar("train/loss", loss.item() * args.grad_accum_steps, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    
                    # Log to wandb
                    if args.use_wandb:
                        wandb.log({
                            "train/loss": loss.item() * args.grad_accum_steps,
                            "train/lr": lr,
                            "train/step": global_step,
                            "train/epoch": epoch
                        })
                
                # Evaluate
                if global_step % args.eval_steps == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    eval_loss, eval_ppl = evaluate(model, val_dataloader, device, args.fp16)
                    
                    logger.info(f"Eval loss: {eval_loss:.4f}, Perplexity: {eval_ppl:.4f}")
                    
                    # Log metrics
                    writer.add_scalar("eval/loss", eval_loss, global_step)
                    writer.add_scalar("eval/perplexity", eval_ppl, global_step)
                    
                    if args.use_wandb:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/perplexity": eval_ppl,
                            "eval/step": global_step
                        })
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(model, optimizer, scheduler, global_step, epoch, args, final=False)
                        logger.info(f"New best model with loss: {best_eval_loss:.4f}")
                    
                    # Switch back to train mode
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch, args, final=False)
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_loss, eval_ppl = evaluate(model, val_dataloader, device, args.fp16)
    logger.info(f"Final eval loss: {eval_loss:.4f}, Perplexity: {eval_ppl:.4f}")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, args.num_epochs-1, args, final=True)
    logger.info("Training completed!")
    
    # Close tensorboard writer
    writer.close()
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
