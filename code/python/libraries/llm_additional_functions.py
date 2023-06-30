import timeit
import requests
from bs4 import BeautifulSoup

#Use the following pieces of context to answer the question at the end. 
#If you don't know the answer, just say that you don't know, don't try to make up an answer.
def format_prompt(question, context) :
    text = f"""
### Assistant: uso el siguiente contexto para responder la pregunta. Si no se la respuesta, solo diré que no la se, no trataré de inventar una respuesta.
Contexto:
{context}
### Human: {question}
### Assistant:"""
    return text.strip()


def get_webpage(url) :
    
    # Send a GET request to the URL and retrieve the HTML content
    response = requests.get(url)
    html_content = response.text

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the main content element where the text is located
    content_element = soup.find(id='mw-content-text')

    # Extract the text from the content element
    text = content_element.get_text()

    # # Save the extracted text into a file
    return text
#%%

# start timer
def tic():
    start_time = timeit.default_timer()
    return start_time

# stop timer
def toc():
    end_time = timeit.default_timer()
    # elapsed_time = end_time - start_time
    # print(f"ET: {elapsed_time:.3f} seconds.")
    return end_time

# %%
