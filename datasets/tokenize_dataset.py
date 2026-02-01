
from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("saved_local_dataset")
tokenizer = AutoTokenizer.from_pretrained("../qwen2.5-coder-7b" ) 

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format("torch", columns=["text","label"])
