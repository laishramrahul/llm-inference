#https://medium.com/@crismunozv/using-fine-tuned-llm-with-vllm-ee34e7db5495
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import numpy as np
from transformers import LlamaTokenizer
from transformers import pipeline
from peft import PeftModel

model_name = "microsoft/phi-2"
peft_model="ms-phi-2-guanaco"
merged_peft_model_name="ms-phi-2-guanaco-merged"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
max_seq_length = tokenizer.model_max_length

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #token="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",
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload")
        
model = PeftModel.from_pretrained(
    model, 
    peft_model, 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 
)

model = model.merge_and_unload()
model.save_pretrained(merged_peft_model_name)
# tokenizer.save_pretrained(merged_peft_model_name)
# model.push_to_hub(merged_peft_model_name)