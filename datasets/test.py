
from transformers import AutoTokenizer, DataCollatorWithPadding

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("../qwen2.5-coder-7b", trust_remote_code=True)

# CRITICAL: Set pad token if missing (fixes your error)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Tokenize without padding (dynamic padding is more efficient)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # Lower to 384 or 256 if you still hit VRAM issues
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Clean columns
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Use dynamic padding via DataCollator (saves VRAM)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# In your Trainer:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,  # This handles padding correctly
    # tokenizer=tokenizer,  # Optional, but good for some features
)
