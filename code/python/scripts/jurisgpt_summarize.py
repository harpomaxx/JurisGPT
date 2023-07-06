from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import UnstructuredODTLoader
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
os.chdir("/home/rodralez/JurisGPT/rawdata/")
loader = UnstructuredODTLoader("13-03850672-7 - Gonzalez Mario.odt")
docs = loader.load()

#%% SPLIT

#%% SPLIT  
# https://www.pinecone.io/learn/chunking-strategies/
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=0)
docs_split = text_splitter.split_documents(docs)

#%% EMBEDDEDING
# Define embedding function
embedding_fnc = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#%% CHROMA
# Non-persistent Chroma
vectordb = Chroma.from_documents(docs_split, embedding_fnc)

#%% QUICK START
# from langchain.docstore.document import Document
# docs = [Document(page_content=t) for t in docs_split[:3]]

from langchain.chains.summarize import load_summarize_chain

prompt_template = """Escribe un resumen en español del siguiente texto:

{text}

RESUMEN CONSISO EN ESPAÑOL:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
chain.run(docs)

#%% 
template = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: {question}

### Assistant:"""

prompt = PromptTemplate(template=template, input_variables=["question"])


text_splitter = CharacterTextSplitter()