from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import pandas as pd
import time
import numpy as np
import tiktoken
import re
import os

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def pdf_to_text(pdf_path):
    """
    Extract text from a PDF file using pdfminer.six with extract_pages.
    Handles issues like missing spaces by processing page by page.
    :param pdf_path: path to the PDF file
    :return: cleaned text extracted from the PDF file
    """
    text = ""  # Initialize the text variable

    # Process each page using extract_pages
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):  # Only process text containers
                page_text = element.get_text()
                # Normalize spaces: split words and rejoin with a single space
                cleaned_text = ' '.join(page_text.split())
                text += cleaned_text + "\n"  # Add a newline between pages

    return text.strip()  # Strip any extra whitespace

def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r'(ACKNOWLEDGMENTS|REFERENCES|AUTHOR INFORMATION|BIBLIOGRAPHY|Acknowledgements)'
    match = re.search(pattern, pdf_text) # Makes it case-insensitive (References is treated as same to REFERENCES)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        end_index = match.end()
        if start_index > len(pdf_text) * 0.5:
            clean_text = pdf_text[:start_index].strip()
        elif end_index > len(pdf_text) * 0.5:
            clean_text = pdf_text[:end_index].strip()
        else:
            clean_text=pdf_text
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5};','\([\d\w]{1,3}\).+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},','\[[\d\w]{1,3}\].+?[\d]{3,5}','[\d\w]{1,3}\).+?[\d]{3,5}\.','[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}','^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text



def load_txt(file_path):
    """
    Load the text from a txt file.
    
    file_path: Path to the txt file to be read (default is "data.txt").
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None