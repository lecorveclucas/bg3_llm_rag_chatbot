import requests
from bs4 import BeautifulSoup
import re
import sys

# Function to split text into chunks of 30 words
def split_text(text, chunk_size=30):
    words = re.findall(r'\S+\s*', text)
    return [''.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# URL of the webpage
url = 'https://baldursgate3.wiki.fextralife.com/Lift+the+Shadow+Curse'
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
    # Check if the next element is of class "bonfire" or "titlearea"
    if 'class' in sibling.attrs and sibling.attrs['class'][0] in ['bonfire', 'titlearea']:
        break
    
# Find all elements with class 'titlearea' or 'bonfire'
extracted_text = page_description + '\n\n'
title_areas = soup.find_all(class_=lambda x: x in ['titlearea', 'bonfire'])

for area in title_areas:
    # Extract the text from the area
    title = area.get_text()
    
    # Find all <p> tags following the area until the next class
    text_elements = []
    for sibling in area.find_next_siblings():
        # print(sibling)
        if sibling.name in ['p']:
            text_elements.append(sibling.get_text())
            
        elif sibling.name in ['ul', 'ol']:
            # Find all <ul> or <ol> tags following the area until the next class
            next_siblings = area.find_next_siblings()
            ul_elements = []
            visited_ul = set()
            for sibling in next_siblings:
                if sibling.name in ['ul', 'ol']:
                    ul_elements.append(sibling)
                elif 'class' in sibling.attrs and sibling.attrs['class'][0] in ['bonfire', 'titlearea']:
                    break

            # Concatenate text from all <ul> elements and their nested <li> elements
            ul_text = ".".join([ul.get_text() for ul in ul_elements])
            ul_text = re.sub(r'[\n\s]+', ' ', ul_text)
            text_elements.append(ul_text)

        elif 'class' in sibling.attrs:
            break

        break
    
    # Combine the text from all paragraphs
    area_text = '\n'.join(text_elements)
    
    # Prepend the title to the extracted text
    full_text = title + ' : ' + area_text
    
    # Split the text into chunks of 30 words
    text_chunks = split_text(full_text)
    
    # Join the chunks with a newline
    formatted_text = '\n'.join(text_chunks)
    
    # Append to the extracted text
    extracted_text += formatted_text + '\n\n'

# Write the chunks to a file with a return line after every chunk
with open('extracted_text.txt', 'w') as file:
    file.write(extracted_text)

print("Text extracted and written to 'extracted_text.txt' file.")