from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredODTLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import NLTKTextSplitter
import os
import sys

# change to project folder
os.chdir("/home/rodralez/JurisGPT/")

# Add the folder path to the sys.path list
sys.path.append('/home/rodralez/JurisGPT/code/python/libraries')

import text_generator_api as tg
import llm_additional_functions as ad

# load document
# Open the 'argentina.txt' file in read mode
# with open('./data/argentina.txt', 'r', encoding='utf-8') as file:
#     # Read the contents of the file
#     documents = file.read()

# # Print the content
# print(documents)

#%% CHROMA
embedding_fnc = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
# embedding_fnc = SentenceTransformerEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

#%% LOAD

# load files in directory
# loader = DirectoryLoader(".", glob="**/*.odt")
# documents = loader.load()
# print(documents)

# load file
loader = UnstructuredFileLoader("./data/babasonicos.txt")
docs = loader.load()

text = docs[0].page_content
# print(text)

#%% SPLIT  

# https://www.pinecone.io/learn/chunking-strategies/
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)

# texts = text_splitter.split_documents(documents)
docs_split = text_splitter.split_documents(docs)

#%% 

# Non-persistent Chroma
db = Chroma.from_documents(docs_split, embedding_fnc, 
                           metadatas=[{"source": str(i)} for i in range(len(docs_split))])

# Save Persistent Chroma
# db = Chroma.from_documents(texts, embedding_fnc, persist_directory="./data/argentina_es_db")
# db.persist()

# Load Persistent Chroma
#db = Chroma(embedding_function=embedding_fnc, persist_directory="./data/argentina_es_db")

#%% 

# question = "How many years has Peronism been in power?"
# question = "What are the main natural resources in Argentina?"
# question = "¿Cuáles son los principales recursos naturales de Argentina?"
# question = '¿Quiénes es el líder de Babasónicos?'
question = '¿Quiénes son los integrantes de Babasónicos?'
#%% QUERY
docs_query = db.similarity_search(question, k=3)
# docs_query = db.max_marginal_relevance_search(question)
context = '\n'.join([doc.page_content for doc in docs_query])
prompt = ad.format_prompt(question, context)

start_time = ad.tic()
response = tg.query_llm(prompt)
end_time = ad.toc()

elapsed_time = end_time - start_time

print ("Human question: ", question)
print ("LLM response: ", response)
print(f"LLM time response: {elapsed_time:.3f} seconds.")

dumb = 0