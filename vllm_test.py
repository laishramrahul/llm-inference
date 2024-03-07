# pip install vllm
from vllm import LLM, SamplingParams
import os
import time
from transformers import pipeline
#os.environ ['CUDA_LAUNCH_BLOCKING'] ='1'import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:22"

prompts = [
    " Cuál es el país más grande del mundo?",
     "Who is Leonardo Da Vinci?",

]
sampling_params = SamplingParams(temperature=0.01, top_p=0.01)
pretrained_model="microsoft/phi-2"
merged_model="ms-phi-2-guanaco-merged"
#llm = LLM(model="huggyllama/llama-7b")
#llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="microsoft/phi-2")
#llm = LLM(model="NousResearch/Llama-2-7b-hf")
#llm = LLM(model="models--NousResearch--Llama-2-7b-hf")
llm = LLM(model=merged_model,tokenizer=pretrained_model)

time1=time.time()
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(time.time()-time1)



prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=merged_model, tokenizer=pretrained_model)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result,"-------------------")
print(result[0]['generated_text'])