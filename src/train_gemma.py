import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path

# Constants
MODEL_ID = "ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 32768  # Using 32K tokens for training to balance memory usage
BATCH_SIZE = 2  # Conservative batch size for 16GB VRAM with longer sequences
GRADIENT_ACCUMULATION_STEPS = 32  # Increased to compensate for smaller batch size
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def prepare_dataset():
    """Prepare the dataset from processed text files."""
    texts = []
    text_dir = Path("data/processed/text_files")
    
    print("Loading text files...")
    for text_file in tqdm(list(text_dir.glob("*.txt"))):
        try:
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()
                if text:
                    # Split into smaller chunks for training
                    words = text.split()
                    chunk_size = MAX_LENGTH * 3  # Approximate words per chunk
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        texts.append({"text": chunk})
        except Exception as e:
            print(f"Error processing {text_file}: {e}")
    
    print(f"Loaded {len(texts)} text chunks from {len(list(text_dir.glob('*.txt')))} files")
    return Dataset.from_list(texts)

def prepare_model():
    """Prepare the GPTQ model with LoRA for training."""
    # Load the quantized model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing dataset...")
    dataset = prepare_dataset()
    
    print("Preparing model...")
    model = prepare_model()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="oracle_model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("oracle_model_final")

if __name__ == "__main__":
    main()
