from transformers import (AutoModelForCausalLM, AutoTokenizer)




model_code = AutoModelForCausalLM.from_pretrained(
        "/root/LLM-models/Llama-2-13b-hf-jurisv2",
        local_files_only= True,
        #quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/root/LLM-models/Llama-2-13b-hf-jurisv2", trust_remote_code=True)

model_code.push_to_hub("Llama-2-13b-hf-juris-adapter-v2")
tokenizer.push_to_hub("Llama-2-13b-hf-juris-adapter-v2")