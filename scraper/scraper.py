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
        
    # Find all elements with class 'titlearea' or 'bonfire'
    extracted_text = page_description + '\n\n'
    title_areas = soup.find_all(class_=lambda x: x in classes_to_parse)


    for area in title_areas:
        # Extract the text from the area
        title = area.get_text()
        
        # Find all <p> tags following the area until the next class
        text_elements = []
        for sibling in area.find_next_siblings():
            # # Case where sibling is empty
            # if not sibling:
            #     continue
            
            # # Case where silbing is a plain text
            # elif sibling.name in ['p']:
            #     text_elements.append(sibling.get_text())
                
            # # Case where silbing is a list
            # elif sibling.name in ['ul', 'ol']:
            #     # Find all <ul> or <ol> tags following the area until the next class
            #     next_siblings = area.find_next_siblings()
            #     ul_elements = []
            #     visited_ul = set()
            #     for sibling in next_siblings:
            #         if not sibling:
            #             continue
            #         elif sibling.name in ['ul', 'ol']:
            #             ul_elements.append(sibling)
            #         elif 'class' in sibling.attrs and sibling.attrs['class'][0] in ['bonfire', 'titlearea']:
            #             break

            #     # Concatenate text from all <ul> elements and their nested <li> elements
            #     ul_text = ".".join([ul.get_text() for ul in ul_elements])
            #     ul_text = re.sub(r'[\n\s]+', ' ', ul_text)
            #     text_elements.append(ul_text)

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
        area_text = '\n'.join(text_elements)
        
        # Prepend the title to the extracted text
        full_text = title + ' : ' + area_text
        
        # # Split the text into chunks of 30 words
        # text_chunks = split_text(full_text)
        
        # # Join the chunks with a newline
        # formatted_text = '\n'.join(text_chunks)
        
        # Append to the extracted text
        extracted_text += full_text + '\n\n'

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
#process_folder(folder_name="quests", classes_to_parse=["titlearea", "bonfire"])

# Process companions
process_folder(folder_name="companions", classes_to_parse=["titlearea", "bonfire", "special"])