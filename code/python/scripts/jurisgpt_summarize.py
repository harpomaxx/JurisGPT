from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
import langchain
from langchain.llms import TextGen
from langchain.text_splitter import NLTKTextSplitter
import os
import yaml

langchain.debug = True

# Define the path to the project folder
PROJECT_PATH = os.path.join(os.getenv('HOME'), 'JurisGPT')
# Change to the project folder using $HOME environment variable
os.chdir(PROJECT_PATH)

import sys
sys.path.append('code/python/libraries')
import jurisgpt_functions as ju

# open the YAML file and load the contents
with open("config/config.yaml", "r") as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)

langchain_api_key = config_data['langchain']['api_key']

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_SESSION"] = "jurisgpt"

#%% DEFINE THE LLM 
model_url = "http://localhost:5000"
llm = TextGen(model_url=model_url)

#%% LOAD
# load file
# change to rawdata folder
# loader = UnstructuredODTLoader("rawdata/fallos/13-03850672-7 - Gonzalez Mario.odt")
loader = UnstructuredFileLoader("rawdata/laboral/10000003219.pdf")
doc = loader.load()
text_raw = doc[0].page_content
text = text_raw.replace("(cid:0)", "")
doc[0].page_content = text
ju.save_text(text_raw, "text_raw.txt")

#%% EXTRACT SECTIONS
titles = ['ANTECEDENTES:', 'SOBRE LA', 'R E S U E L V E:']
# titles = ['R E S U E L V E:']

sections = ju.extract_sections(text, titles)

ju.save_text(sections[0], "tmp.txt")

# Create a new instance of UnstructuredFileLoader
loader = UnstructuredFileLoader("tmp.txt")
# Load the file and assign it to the doc variable
doc = loader.load()

#%% QUICK START

from langchain.chains.summarize import load_summarize_chain

prompt_template = """
your tasks are:
1. translate this text into english
2. summarize the text
3. translate the summary into spanish

text:
{text}

Summarized text in Spanish:
"""

# PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
# chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
# chain.run(docs)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True, return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
chain({"input_documents": doc}, return_only_outputs=True)

# chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True)
# chain({"input_documents": doc_split}, return_only_outputs=True)

dump_break = 0