import chromadb
import sentence_transformers

from datasets import load_dataset, Dataset
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline, set_seed
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from rouge import Rouge

import torch
import os
import re
import pandas as pd  # Don't forget to import pandas
import yaml
    
def print_rouge_table(all_scores):
    pd.set_option('display.max_rows', None)
    
    system_ids = []
    metrics = []
    rs = []
    ps = []
    fs = []

    # Loop through each dictionary to extract the values
    for score_dict in all_scores:
        system_id = score_dict['system_id']

        for metric, values in score_dict.items():
            if metric == 'system_id':
                continue  # Skip the system_id, we've already saved it

            system_ids.append(system_id)
            metrics.append(metric)
            rs.append(values['r'])
            ps.append(values['p'])
            fs.append(values['f'])

    # Create the DataFrame
    df_long = pd.DataFrame({
        'system_id': system_ids,
        'metric': metrics,
        'r': rs,
        'p': ps,
        'f': fs
    })

    return df_long

def calculate_rouge(system_dir, model_dir, system_filename_pattern, model_filename_pattern):
    # Initialize Rouge
    rouge = Rouge()
    
    # Compile the regular expressions for filename patterns
    system_re = re.compile(system_filename_pattern)
    model_re = re.compile(model_filename_pattern)
    
    # Initialize storage for scores
    all_scores = []

    # Loop through the files in the system summaries directory
    for system_filename in os.listdir(system_dir):
        match = system_re.match(system_filename)
        if not match:
            continue

        # Extract ID
        system_id = match.groups()[0]

        # Find the corresponding model summary file using ID
        model_filename = f"{system_id}_summary.txt"
        model_path = os.path.join(model_dir, model_filename)

        # Check if model summary exists
        if not os.path.exists(model_path):
            print(f"Model summary for {system_id} not found. Attempted to open {model_path}")
            continue

        

        # Check if model summary exists
        if not os.path.exists(model_path):
            print(f"Model summary for {system_id} not found.")
            continue

        # Read system and model summaries
        with open(os.path.join(system_dir, system_filename), 'r', encoding='utf-8') as f:
            system_summary = f.read().strip()
            system_summary = re.sub('<.*?>', '', system_summary)
            #print(system_summary)
        
        with open(model_path, 'r', encoding='utf-8') as f:
            model_summary = f.read().strip()
    
        # Compute the ROUGE score
        score = rouge.get_scores(system_summary, model_summary)[0]
        #all_scores.append(score)
        all_scores.append({
            'system_id': system_id,
            'rouge-1': score['rouge-1'],
            'rouge-2': score['rouge-2'],
            'rouge-l': score.get('rouge-l', {})  # Add this line to include ROUGE-L
        })

    return all_scores

def create_summary_es(rule_text, embedding, llm, queries): 
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(rule_text)
    db_hf = Chroma.from_texts(texts, embedding)
    retriever = db_hf.as_retriever()
    index = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever,
                                 #return_source_documents=True
                                )
    print("[] Summarizing ant")
    summary_ant = (index.run(queries['summary_ant']))
    print("[] Summarizing 1")
    summary_1 = (index.run(queries['summary_1']))
    print("[] Summarizing 2")
    summary_2 = (index.run(queries['summary_2']))
    print("[] Summarizing 3")
    summary_3 = (index.run(queries['summary_3']))
    print("[] Summarizing res")  
    summary_res = (index.run(queries['summary_res']))
    
    prompt =f"""
<s>[INST]<<SYS>>
Think after answer. Your answer should be in spanish. Not in english.
<</SYS>> 
Eres un oficial perito de la corte suprema de Mendoza en Argentina. A partir del siguiente resumen de una sentencia de la corte :
```
{summary_ant}
{summary_1}
{summary_2}
{summary_3}
{summary_res}
```
Debes escribir un sumario en donde se expliquen las causas detras de la decision final de la sentencia. 

El sumario de la corte no puede contener mas de 700 palabras.
No deben mencionarse NUNCA:
    1. Los nombres de las cuestiones como ser: primera cuestion, segunda cuestion o tercera cuestion.
    2. Los nombres de los casos.
    3. Los nombres de los ministros de la corte.
    4. Las fechas y la ubicacion de la corte.
    5. La palabra corte
El sumario tener en cuenta que este puede ayudar en futura jurisprudencia. 
Solo en caso de disidencia se debe indicar el nombre del  ministro de la corte.
[/INST]

"""
    index.retriever.vectorstore.delete_collection()
    print("[] Executing final prompt")
    print(prompt)
    res = llm(prompt)
    return res


def create_summary_from_dataset(dataset, output_directory_path, embedding, llm, queries):
    import time
    # Loop through each file in the directory
    # for i in range(len(dataset['texto'])):
    for i in range(0,10):
            print(f"working on {dataset['fallo'][i]}")
            # Generate summary
            start_time = time.time()
            summary = create_summary_es(dataset['sentencia'][i], embedding, llm, queries)
            stop_time = time.time()  
            # Save summary to new file
            summary_filename = f"{dataset['fallo'][i]}_summary.txt"
            summary_file_path = os.path.join(f"{output_directory_path}/model/", summary_filename)  
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f'created {summary_file_path} in {stop_time - start_time} seconds')
             # Save summary to new file
            summary_filename = f"{dataset['fallo'][i]}_gold_summary.txt"
            summary_file_path = os.path.join(f"{output_directory_path}/system/", summary_filename)  
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                f.write(dataset['texto'][i])
            
            
if __name__ == "__main__" :
    
    model_path = "/root/LLM-models/Llama-2-13b-hf-jurisv3/"
    tokenizer_path = "/root/LLM-models/Llama-2-13b-hf-jurisv3/"
    
    #model_path = "harpomaxx/Llama-2-13b-hf-juris-adapter"
    #tokenizer_path = "harpomaxx/Llama-2-13b-hf-juris-adapter"
    
    
    #model_path = "clibrain/lince-mistral-7b-it-es"
    #tokenizer_path = "clibrain/lince-mistral-7b-it-es"
    
    output_directory_path = "/root/JurisGPT/data/laboral/sumariosbigdb/"
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype= getattr(torch, "float16"),
    bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map ="auto",
                #device_map={"": 0},
                trust_remote_code=True)


    tok = AutoTokenizer.from_pretrained(tokenizer_path,
                trust_remote_code=True,
                device_map= "auto"
                )

    generator = pipeline('text-generation',
                             model=model,
                             tokenizer=tok,
                             #streamer=streamer,
                             return_full_text = False,
                             do_sample = True,
                             temperature = 0.1,
                             top_k = 2,
                             no_repeat_ngram_size = 3,
                             max_new_tokens = 500 )

    embedding = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L12-v2")
    llm = HuggingFacePipeline(pipeline = generator)

     

    # ### Load Dataset
    print("[] load dataset.")
    val_dataset = load_dataset("harpomaxx/jurisgpt", split="test")
    
    with open("prompts.yml", 'r') as file:
        prompts = yaml.safe_load(file)
    
    create_summary_from_dataset(val_dataset, output_directory_path, embedding = embedding, llm = llm, queries= prompts)