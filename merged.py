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
model.merge_and_unload("laishram/bloom-560m-merged-lora-tagger")
model.save_pretrained("./bloom-560m-merged-lora-tagger")
model.push_to_hub("bloom-560m-merged-lora-tagger")