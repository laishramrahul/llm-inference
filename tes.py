import re
from transformers import TextStreamer
import torch
import random
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
peft_model_path="./model/peft-question-generation_ver2_result"
peft_model_dir = "peft-question-generation_ver2_result"
import time
#custom_cache_directory = "./models"
import nltk
nltk.download('punkt')
start_time = time.time()
# load base LLM model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)
#trained_model.push("test_qg")