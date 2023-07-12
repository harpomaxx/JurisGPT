import timeit
import requests
from bs4 import BeautifulSoup

#%% Extract the sections of the text that correspond to the titles    
def extract_sections(text, titles):
    """
    Extracts sections from a text based on specified titles.

    Args:
        text (str): The text to extract sections from.
        titles (list of str): A list of titles to identify the start of each section.

    Returns:
        list: A list of sections extracted from the text.

    Example:
        >>> text = "Introduction\n\nThis is the introduction.\n\nMethods\n\nHere are the methods:\n\nMethod 1\nMethod 2"
        >>> titles = ["Introduction", "Methods"]
        >>> extract_sections(text, titles)
        ['Introduction\n\nThis is the introduction.', 'Methods\n\nHere are the methods:\n\nMethod 1\nMethod 2']
    """
    sections = []
    lines = text.split('\n')
    current_section = None
    for line in lines:
        line = line.strip()
        if any(title in line for title in titles):
            if current_section is not None:
                sections.append(current_section)
            current_section = line
        elif current_section is not None:
            current_section += ' ' + line

    if current_section is not None:
        sections.append(current_section)

    return sections

#%%
def prompt_summary(instruction, text):
    """
    Formats a Spanish prompt by combining a summarization instruction and text.

    Args:
        instruction (str): The instruction to be included in the prompt.
        text (str): The text to be included in the prompt.

    Returns:
        str: The formatted prompt.

    Example:
        >>> format_prompt('Summarize the following text.', 'Paris is the capital of France')
    """
    text = f"""
### Humano: {instruction}
Texto:
{text}
### Resumen:"""
    return text.strip()

#%%
def prompt_simple(instruction, question):
    """
    Formats a prompt by combining a instruction and a question.

    Args:
        instruction (str): The instruction to be included in the prompt.
        question (str): The question to be included in the prompt.

    Returns:
        str: The formatted prompt.

    Example:
        >>> simple_prompt('A chat between a curious user and an artificial intelligence assistant. \ 
        The assistant gives helpful, detailed, and polite answers to the user's questions.\n', 'What is the capital of France?')
    """
    text = f"""
{instruction}
### Human: {question}
### Assistant:"""
    return text.strip()


#%%
def prompt_context(instruction, question, context):
    """
    Formats a Spanish prompt by combining a instruction, context, and question.

    Args:
        instruction (str): The instruction to be included in the prompt.
        question (str): The question to be included in the prompt.
        context (str): The context to be included in the prompt.

    Returns:
        str: The formatted prompt.

    Example:
        >>> format_prompt('A chat between a curious user and an artificial intelligence assistant. \ 
        The assistant gives helpful, detailed, and polite answers to the user's questions.', 
        'Paris is the capital of France', What is the capital of France?')
    """
    text = f"""
{instruction}
Contexto:
{context}
### Human: {question}
### Assistant:"""
    return text.strip()


#%%
def get_webpage(url) :
    """
    Retrieves the text content of a web page by sending a GET request to the specified URL.

    Args:
        url (str): The URL of the web page to retrieve.

    Returns:
        str: The text content of the web page.

    Raises:
        requests.exceptions.RequestException: If an error occurs while sending the GET request.

    Example:
        >>> get_webpage("https://www.example.com")
        'Welcome to Example.com!'

    Note:
        This function requires the `requests` and `beautifulsoup4` packages to be installed.
    """

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
    return end_time

# %%
def save_text(text_raw, file_path):

    with open(file_path, 'w') as file:
        file.write(text_raw)