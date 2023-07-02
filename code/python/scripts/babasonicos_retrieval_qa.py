'''
## Retrieval-based QA System for Babasónicos Wikipedia Page

This script implements a retrieval-based question-answering (QA) system using the information from the Babasónicos Wikipedia page. Babasónicos is a rock band from Argentina, and the script utilizes various technologies to provide answers to user queries. The script makes use of the following technologies:

1. langchain: A framework for developing applications powered by language models.
2. SentenceTransformer: A library for generating sentence embeddings.
3. NLTKTextSplitter: A tool for splitting the text into smaller chunks.
4. Chroma DB: A database for storing the text chunks and their corresponding embeddings.
5. Text Generator UI: A user interface for interacting with the language model.

To use this script, ensure that the required dependencies are installed. The script is designed to retrieve information specifically from the Babasónicos Wikipedia page. Queries can be entered via the Text Generator UI, and the script will provide relevant answers based on the available information.

Please note that the LLM (Large Language Model) has no prior knowledge about Babasónicos or its Wikipedia page. All answers are generated based on the information contained within the page itself.

For further assistance or inquiries, refer to the script documentation or contact the script maintainer.
'''

# Before running this script, you have to run the following commands in the terminal:
# ~$ pip install -r $PROJECT_PATH/requirements.txt
# ~$ cd text-generation-webui/
# ~$ python server.py --verbose --api
# Then, load the Vicuna 13B model in the UI.

from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import NLTKTextSplitter
import os
import sys

# Define the path to the project folder
PROJECT_PATH = os.path.join(os.getenv('HOME'), 'JurisGPT')

# Change to the project folder using $HOME environment variable
os.chdir(PROJECT_PATH)

# Add the library folder path to the sys.path list
sys.path.append(PROJECT_PATH + '/code/python/libraries')

import text_generator_api as tg
import llm_additional_functions as ad

#%% EMBEDDEDING
# Define embedding function
embedding_fnc = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#%% LOAD DATA
# load data file
loader = UnstructuredFileLoader("./data/babasonicos.txt")
docs = loader.load()

#%% SPLIT DOCUMENTS IN CHUNKS 
# https://www.pinecone.io/learn/chunking-strategies/
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)

docs_split = text_splitter.split_documents(docs)

#%% CHROMA DB
# Non-persistent Chroma
db = Chroma.from_documents(docs_split, embedding_fnc)

# Save Persistent Chroma
# db = Chroma.from_documents(texts, embedding_fnc, persist_directory="./data/argentina_es_db")
# db.persist()

# Load Persistent Chroma
#db = Chroma(embedding_function=embedding_fnc, persist_directory="./data/argentina_es_db")

#%% QUESTION
# question = '¿Quiénes es el líder de Babasónicos?'
question = '¿En qué año se fundó la banda Babasónicos?'
# question = '¿Quiénes son los integrantes de Babasónicos?'

#%% QUERY SIMILAR CHUNKS
docs_query = db.similarity_search(question, k=3)

#%% CONTEXT
context = '\n'.join([doc.page_content for doc in docs_query])

#%% PROMPT
# header_en = "### Assistant: I use the following context to answer the question. If I don't know the answer, I'll simply say that I don't know it, I won't try to invent an answer."
header = "### Assistant: uso el siguiente contexto para responder la pregunta. Si no se la respuesta, solo diré que no la se, no trataré de inventar una respuesta."
prompt = ad.format_prompt(header, question, context)

#%% LLM QUERY
print ("Asking a question to LLM...")

start_time = ad.tic()
response = tg.query_llm(prompt)
end_time = ad.toc()

elapsed_time = end_time - start_time

print ("Human question: ", question)
print ("LLM response: ", response)
print(f"LLM response time: {elapsed_time:.3f} seconds.")

dumb_break = 0