import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split
from pynvml import *
import pandas as pd
import random
#import mlflow
import os
#os.environ["WANDB_DISABLED"] = "true"


## CONFIGURATION
#
dataset_path = "/root/JurisGPT/rawdata/laboral/sumariosbigdb/sumariosdb.json" # in JSON
model_path = "meta-llama/Llama-2-13b-chat-hf"
tokenizer_path = "meta-llama/Llama-2-13b-chat-hf"
trained_model_checkpoints_save_path = "/root/checkpoints/"
trained_model_save_path = "/root/LLM-models/Llama-2-13b-hf-jurisv2"
#mlflow_experiment = "/llama-2-7b-guanaco"

#mlflow.set_tracking_uri("http://147.32.83.60")
#mlflow.set_experiment(mlflow_experiment)



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

# ### Bits and Bytes configuration for using quantization
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
print("[] load model.")
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map= "auto",
        #device_map={"": 0},
        trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, 
                                          trust_remote_code=True)
#Note: the tokenizer from the llama-2 models does not use a pad token therefore the following line is necessary:
tokenizer.pad_token = tokenizer.eos_token 
## TRAINING

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.5,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj","k_proj"] # obtained by the output of the model
)

model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir= trained_model_checkpoints_save_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    optim='adamw_bnb_8bit',
    save_steps=500,
    fp16=True,
    #report_to='mlflow',
    #report_to='none',
    
    logging_steps=100,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    #max_steps=5000,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps"  
)

# ### Load Dataset
print("[] load dataset.")
train_dataset = load_dataset("harpomaxx/jurisgpt", split="train")
#train_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
val_dataset = load_dataset("harpomaxx/jurisgpt", split="test")


#dataset = pd.read_json(dataset_path)
#train_dataset, val_dataset = train_test_split(dataset, test_size=0.20, random_state=42)

#train_dataset = Dataset.from_pandas(train_dataset)
#val_dataset = Dataset.from_pandas(val_dataset)

# ### Format data for training

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['texto'])):
      
        #text = f"Voces: {example['voces'][i]}\nSumario: {example['texto'][i]}"
        text =f"""<s>[INST]<<SYS>>
        Think after answer. Your answer should be in spanish. Not in english.
        <</SYS>> 
        Eres un oficial perito de la corte suprema de Mendoza en Argentina. A partir del siguiente resumen de una sentencia de la corte :
        ```
        {example['resumen'][i]}
        ```
        Debes escribir un sumario en donde se expliquen las causas detras de la decision final de la sentencia. 
        Escribir el sumario de la corte en no mas de 700 palabras.
        Favor de no mencionar NUNCA:
            1. Los nombres de las secciones como ser: primera cuestion, segunda cuestion o tercera cuestion.
            2. Los nombres de los casos.
            3. Los nombres de los ministros de la corte.
            4. Las fechas y la ubicacion de la corte.
            5. La palabra corte
            Cuando escribas el sumario tener en cuenta que este puede ayudar en futura jurisprudencia. 
            En caso de disidencia indicar el nombre del  ministro de la corte. 
        
        Escribe tu respuesta en un español correcto.
        [/INST]
        {example['texto'][i]}</s>
        """
        output_texts.append(text)
    return output_texts

instruction_template = "[INST]"
response_template = "[/INST]"

collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                           response_template=response_template, 
                                           tokenizer=tokenizer, 
                                           mlm=False
                                           )
print("[] Training.")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    #dataset_text_field="text",
    formatting_func= formatting_prompts_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    #data_collator= collator,
    
)

#with mlflow.start_run() as run:

trainer.train()
print(f"[] Saving LoRa at {trained_model_save_path}/lora")
model.save_pretrained(f'{trained_model_save_path}/lora')
# Free memory for merging weights
del model
torch.cuda.empty_cache()
##  code for merging
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        #device_map={"": 0}, # for setting up a particular GPU
        device_map="auto", # for multiple GPUs
        trust_remote_code=True
)
#merged_model = model.merge_and_unload() #Not supporting 8bit models
model = PeftModel.from_pretrained(model,f'{trained_model_save_path}/lora',local_files_only=True)
print(f"[] Saving Model at {trained_model_save_path}")
model.save_pretrained(trained_model_save_path,safe_serialization=True)
print(f"[] Saving Tokenizer at {trained_model_save_path}")
tokenizer.save_pretrained(trained_model_save_path,safe_serialization=True)

## TODO:
## Add some parameters