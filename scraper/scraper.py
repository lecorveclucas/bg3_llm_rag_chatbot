import requests
from bs4 import BeautifulSoup
import re
import sys
import os

# Function to split text into chunks of 30 words
def split_text(text, chunk_size=30):
    words = re.findall(r'\S+\s*', text)
    return [''.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_content(url: str, output_file_path: str, classes_to_parse: list):
    title = url.split("/")[-1].replace("+", " ")

    # Send a GET request to the webpage
    response = requests.get(url)

    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.text, 'html.parser')

    page_description = title + " : "

    # Get first text elements that describe the page
    p_tags = soup.find_all('p')
    text_elements = []

    # Iterate through <p> tags
    for p_tag in p_tags:
        # Extract text from <p> tag
        page_description += p_tag.get_text() + ' '
        sibling = p_tag.find_next_sibling()
        if not sibling:
            continue
        # Check if the next element is of class "bonfire" or "titlearea"
        if 'class' in sibling.attrs and sibling.attrs['class'][0] in classes_to_parse:
            break

    # Check if page has been found
    if "page not found" in page_description.lower():
        return None
        
    # Find all elements with class 'titlearea' or 'bonfire'
    extracted_text = page_description + '\n\n'
    title_areas = soup.find_all(class_=lambda x: x in classes_to_parse)


    for area in title_areas:
        # Extract the text from the area
        title = area.get_text()
        
        # Find all <p> tags following the area until the next class
        text_elements = []
        for sibling in area.find_next_siblings():
            # Case where sibling is empty or the next class
            if not sibling or ('class' in sibling.attrs and sibling.attrs['class'][0] in ['bonfire', 'titlearea']):
                break
            
            # Case where sibling is a plain text
            elif sibling.name in ['p']:
                text_elements.append(sibling.get_text())
            
            # Case where sibling is a list
            elif sibling.name in ['ul', 'ol']:
                # Concatenate text from all <ul> or <ol> elements and their nested <li> elements
                list_text = '.'.join([li.get_text() for li in sibling.find_all('li')])
                list_text = re.sub(r'[\n\s]+', ' ', list_text)
                text_elements.append(list_text)
            
            # Case where sibling is the next class
            elif 'class' in sibling.attrs:
                break
           
        # Combine the text from all paragraphs
        area_text = ' '.join(text_elements)
        
        # Prepend the title to the extracted text
        full_text = title + ' : ' + area_text
        
        # # Split the text into chunks of 30 words
        # text_chunks = split_text(full_text)
        
        # # Join the chunks with a newline
        # formatted_text = '\n'.join(text_chunks)
        
        # Append to the extracted text
        extracted_text += full_text + '\n\n' if area != title_areas[-1] else full_text

    # Clean text
    extracted_text = (extracted_text
                      .replace("...", " ")
                      .replace("..", ".")
                      .replace("\xa0", " ")
                      .replace("Completing Quests allows players to learn more about the world and characters in Baldur's Gate 3, as well as earning more loot and experience to become more powerful.", "")
                      .replace("Completing Quests allows players to learn more about the world and characters in Baldur's Gate 3, as well as earn more loot and experience to become more powerful.", "")
    )
    
    # Write the chunks to a file with a return line after every chunk
    with open(output_file_path, 'w') as file:
        file.write(extracted_text)

    print("Text extracted and written to output_file_path file.\n")

# Process quests
def process_folder(folder_name: str, classes_to_parse: list):
    output_folder = os.path.join("data", folder_name)
    names_list_file = os.path.join("scraper", folder_name, folder_name +"_list.txt")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(names_list_file, "r") as f:
        names_list = f.readlines()
    
    for name in names_list:
        name = name.replace("\n", "").replace(" ", "+")
        name_url = "https://baldursgate3.wiki.fextralife.com/" + name
        print(name_url, "\n")
        output_file_name = name.replace("+", "_") + ".txt"
        extract_content(url=name_url, output_file_path=os.path.join(output_folder, output_file_name), classes_to_parse=classes_to_parse)

# Process quests
# process_folder(folder_name="quests", classes_to_parse=["titlearea", "bonfire"])

# Process companions
# process_folder(folder_name="companions", classes_to_parse=["titlearea", "bonfire", "special"])

# Process races
# process_folder(folder_name="races", classes_to_parse=["titlearea", "bonfire", "special"])

# Process bosses
# process_folder(folder_name="bosses", classes_to_parse=["titlearea", "bonfire", "special"])

# Process npcs
# process_folder(folder_name="npcs", classes_to_parse=["titlearea", "bonfire", "special"])




import os
import streamlit as st
import glob
import base64
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from transformers import AutoTokenizer


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf',
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',  # Q6_K was used too but quite slow
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=512,
    # Context size
    context_window=8192, # Had to increase form 4096 to 8192 because the evaluation create 4600 tokens prompt
    # kwargs to pass to __call__()
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 10},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)
