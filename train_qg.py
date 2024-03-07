import torch
import time
import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
import random
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

custom_cache_directory = "./models"
from datasets import *
ds = load_dataset("csv", data_files=["tamil_filtered.csv"],
         header=None, names=['Instruction','context', 'Output'])
#ds.train_test_split(test_size=0.1)

#dataset
train_testvalid = ds['train'].train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

def format_instruction(Instruction:str,INPUT: str, output: str):
	return f"""### Instruction:
{Instruction}
### Input:
{INPUT}

### Question:
{output}
"""

def generate_instruction_dataset(data_point):

    return {

        "INPUT": data_point["context"],
        "output": data_point["Output"],
        "Instruction":data_point["Instruction"],
        "text": format_instruction(data_point["Instruction"],data_point["context"],data_point["Output"])
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
#        .map(generate_instruction_dataset).remove_columns(['__index_level_0__'])
                .map(generate_instruction_dataset)

    )

# sample_dataset = dataset.filter(lambda example, index: index % 100 == 0, with_indices=True)

# sample_dataset["train"] = process_dataset(sample_dataset["train"])
# sample_dataset["test"] = process_dataset(sample_dataset["validation"])
# sample_dataset["validation"] = process_dataset(sample_dataset["validation"])

## APPLYING PREPROCESSING ON WHOLE DATASET
ds["train"] = process_dataset(ds["train"])
ds["test"] = process_dataset(ds["test"])
ds["valid"] = process_dataset(ds["valid"])
# Select 1000 rows from the training split
#train_data = ds['train'].shuffle(seed=42).select([i for i in range(10000)])
train_data = ds['train'].shuffle(seed=42)

# Select 100 rows from the test and validation splits
#test_data = ds['test'].shuffle(seed=42).select([i for i in range(500)])
#validation_data = ds['valid'].shuffle(seed=42).select([i for i in range(500)])
test_data = ds['test'].shuffle(seed=42)
validation_data = ds['valid'].shuffle(seed=42)

train_data,test_data,validation_data
train_data['text']

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id =  "NousResearch/Llama-2-7b-hf"
# model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", cache_dir=custom_cache_directory)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=custom_cache_directory)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # target_modules=["query_key_value"],
    target_modules=["q_proj", "k_proj","v_proj"], #specific to Llama models.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

OUTPUT_DIR = "llama2-docsum-adapter_updated_domain"

# %load_ext tensorboard
# %tensorboard --logdir llama2-docsum-adapter/runs

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=validation_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

# peft_model_path="./peft-dialogue-summary_updated_domain"
# trainer.model.save_pretrained(peft_model_path)
# tokenizer.save_pretrained(peft_model_path)
model.push_to_hub("peft_test_qg")