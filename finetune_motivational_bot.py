import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import os

# Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

# Load dataset
dataset = load_dataset("json", data_files="motivation_data.jsonl", split="train")

# Preprocess
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = inputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
args = TrainingArguments(
    output_dir="./lora-motivational-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True if torch.cuda.is_available() else False,
    fp16=True if not torch.cuda.is_available() else False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Save model
model.save_pretrained("./lora-motivational-model")
tokenizer.save_pretrained("./lora-motivational-model")