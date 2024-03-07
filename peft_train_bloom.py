#Setup the model
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", 
    load_in_8bit=True, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")


#Freezing the original weights
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

#Setting up the LoRa Adapters
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

from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

#data
import transformers
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")

def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example

data['train'] = data['train'].map(merge_columns)
data['train']["prediction"][:5]

data['train'][0]

data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

# training

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=2,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model = model.save_pretrained("laishram/bloom-560m-lora-tagger")
# model.push_to_hub("laishram/bloom-560m-lora-merged-tagger")


from peft import AutoPeftModel,AutoPeftModelForCausalLM
# args=transformers.TrainingArguments(
#         per_device_train_batch_size=2, 
#         gradient_accumulation_steps=2,
#         warmup_steps=100, 
#         max_steps=200, 
#         learning_rate=2e-4, 
#         fp16=True,
#         logging_steps=1, 
#         output_dir='outputs'
#     )  
basemodel="bigscience/bloom-560m"
adaptername="laishram/bloom-560m-lora-tagger"
model = AutoPeftModelForCausalLM.from_pretrained(adaptername)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload("laishram/bloom-560m-merged-lora-tagger")