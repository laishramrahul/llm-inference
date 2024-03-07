#https://medium.com/@crismunozv/using-fine-tuned-llm-with-vllm-ee34e7db5495
import torch
import numpy as np
import pandas as pd
import json
import os
from vllm import LLM, SamplingParams
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

os.environ['CUDA_VISIBLE_DEVICES']="0"
base_model_name="microsoft/phi-2"
merged_peft_model_name="ms-phi-2-guanaco-merged"
llm = LLM(model=merged_peft_model_name, tokenizer=base_model_name)

output = llm.generate("what is ai")
print(output)