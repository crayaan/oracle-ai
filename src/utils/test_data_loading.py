import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add the parent directory to the path so we can import from train_gemma
sys.path.append(str(Path(__file__).parent.parent))
from train_gemma import prepare_dataset, MODEL_ID, MAX_LENGTH

def test_data_loading():
    """Test the data loading and tokenization process."""
    print("\n=== Testing Data Loading ===")
    
    # Test dataset preparation
    print("\n1. Testing Dataset Creation:")
    try:
        dataset = prepare_dataset()
        print(f"   ✓ Successfully created dataset with {len(dataset)} examples")
        
        # Sample and print a few examples
        print("\n2. Sample Data Preview:")
        for i in range(min(3, len(dataset))):
            text = dataset[i]["text"]
            print(f"\n   Sample {i+1} preview (first 100 chars):")
            print(f"   {text[:100]}...")
            print(f"   Length (chars): {len(text)}")
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {str(e)}")
        return
    
    # Test tokenization
    print("\n3. Testing Tokenization:")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   Testing tokenization of {min(5, len(dataset))} samples...")
        for i in tqdm(range(min(5, len(dataset))), desc="   Processing"):
            text = dataset[i]["text"]
            tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
            token_length = len(tokens["input_ids"])
            print(f"   Sample {i+1} token length: {token_length}")
            
            if token_length >= MAX_LENGTH:
                print(f"   ⚠️  Warning: Sample {i+1} was truncated to {MAX_LENGTH} tokens")
    except Exception as e:
        print(f"   ✗ Tokenization test failed: {str(e)}")
        return
    
    # Memory usage estimation
    print("\n4. Memory Usage Estimation:")
    try:
        # Estimate memory usage for full dataset
        avg_token_length = sum(len(tokenizer(dataset[i]["text"], truncation=True, max_length=MAX_LENGTH)["input_ids"]) 
                             for i in range(min(5, len(dataset)))) / min(5, len(dataset))
        estimated_tokens = avg_token_length * len(dataset)
        estimated_memory_gb = (estimated_tokens * 2 * 4) / (1024 ** 3)  # Assuming 2 bytes per token and 4 for gradients
        print(f"   Estimated average tokens per sample: {avg_token_length:.0f}")
        print(f"   Estimated total tokens: {estimated_tokens:,.0f}")
        print(f"   Estimated memory usage during training: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 14:  # Leaving some buffer for the 16GB GPU
            print("   ⚠️  Warning: Estimated memory usage is high, consider reducing batch size or gradient accumulation")
    except Exception as e:
        print(f"   ✗ Memory estimation failed: {str(e)}")
    
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    test_data_loading() 