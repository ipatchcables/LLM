from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load your dataset
dataset = load_from_disk("saved_local_dataset")  # Assuming it has "train" and "test" splits

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("../qwen2.5-coder-7b", trust_remote_code=True)
# Qwen models often don't have a pad token — add one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,              # Keep reasonable; lower to 256 if still OOM
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove text, rename label, set format
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
#tokenized_dataset.set_format("torch")

# 3. 4-bit quantization config (huge VRAM saver)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
    bnb_4bit_use_double_quant=True,
)

# 4. Load quantized base model for classification
base_model = AutoModelForSequenceClassification.from_pretrained(
    "../qwen2.5-coder-7b",
    num_labels=2,
    quantization_config=quant_config,
    device_map="auto",  # Automatically places layers on GPU
)

# Prepare for QLoRA
base_model = prepare_model_for_kbit_training(base_model)

# 5. LoRA config (tunes only ~1% of params)
lora_config = LoraConfig(
    r=64,               # Rank — 32–128 common; lower = less VRAM
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Typical for Qwen
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)

# ower batch size  for 12GB VRAM
training_args = TrainingArguments(
    output_dir="./my_model_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,     # Start low (2–8); increase if possible
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,     # Effective batch = 4 * 4 = 16 (or higher)
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,                # Higher LR common for LoRA/QLoRA (1e-4 to 5e-4)
    fp16=True,                         # Or bf16=True if your GPU supports it (Ampere+)
    optim="paged_adamw_8bit",          # Memory-efficient optimizer
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
 #   report_to="none",                  # Disable if you don't use wandb/tensorboard
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,   # ← Add this!
)

# 8. Train!
trainer.train()

# After training, save the adapter
model.save_pretrained("./my_lora_adapter")
# To merge and use later: peft_model.merge_and_unload()
