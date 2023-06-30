from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredODTLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import NLTKTextSplitter
import os
import sys

# Before running this script, you have to run the following commands in the terminal:
# ~$ pip install -r $PROJECT_PATH/requirements.txt
# ~$ cd text-generation-webui/
# ~$ python server.py --verbose --api

# Define the path to the project folder
PROJECT_PATH = os.path.join(os.getenv('HOME'), 'JurisGPT')

# Change to the project folder using $HOME environment variable
os.chdir(PROJECT_PATH)

# Add the folder path to the sys.path list
sys.path.append(PROJECT_PATH + '/code/python/libraries')

import text_generator_api as tg
import llm_additional_functions as ad

#%% EMBEDDEDING

# Define embedding function
embedding_fnc = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

#%% LOAD DATA

# load file
loader = UnstructuredFileLoader("./data/babasonicos.txt")
docs = loader.load()

#%% SPLIT DOCUMENTS IN CHUNKS 

# https://www.pinecone.io/learn/chunking-strategies/
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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

question = '¿Quiénes es el líder de Babasónicos?'
# question = '¿Quiénes son los integrantes de Babasónicos?'

#%% QUERY SIMILAR CHUNKS

docs_query = db.similarity_search(question, k=3)
# docs_query = db.max_marginal_relevance_search(question)

#%% CONTEXT

context = '\n'.join([doc.page_content for doc in docs_query])

#%% PROMPT
header = "### Assistant: uso el siguiente contexto para responder la pregunta. Si no se la respuesta, solo diré que no la se, no trataré de inventar una respuesta."
prompt = ad.format_prompt(header, question, context)

#%% QUERY LLM

print ("Asking a question to LLM...")

start_time = ad.tic()
response = tg.query_llm(prompt)
end_time = ad.toc()

elapsed_time = end_time - start_time

print ("Human question: ", question)
print ("LLM response: ", response)
print(f"LLM time response: {elapsed_time:.3f} seconds.")

dumb = 0