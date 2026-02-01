# Using an Instruction-Tuned Code Model
# Download model locallly
# huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./qwen2.5-coder-7b

import torch 

from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig



# Local model path
local_model_path = "./qwen2.5-coder-7b"  

# VRAM to ~5-6GB on a 12GB GPU)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=quantization_config,      # This enables 4-bit loading
    device_map="auto",                            # Automatically places layers on GPU/CPU
    trust_remote_code=True
)

model.eval()  # Set to evaluation mode

# Example buggy code
buggy_code = """
	sets.Weapons.['Almace'] = {
		main="Almace",
		sub={ name="Sakpata's Sword", augments={'Path: A',}},
	}
"""

# Prompt
messages = [
    {"role": "system", "content": "You are an expert Lua coder specifically for Final Fantasy XI (an online MMORPG). Fix bugs, improve the code, and explain your changes clearly."},
    {"role": "user", "content": f"Fix the following code:\n\n{buggy_code}"}
]

# Chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():  # Saves memory during inference
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1  
    )

# Decode only the new generated part (excluding the input prompt)
generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(response)