import os
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import tiktoken
from AI_agents.tools.data_mining.utils import pdf_to_text, load_txt, remove_ref
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunkfiles:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.enc = tiktoken.get_encoding("cl100k_base")
        
    # Read all files in the prepared folder    
    def read_files_list(self):
        file_list=[]
        if self.path is None:
            raise ValueError("Path cannot be None")
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name)  # Create the full path
            if os.path.isfile(file_path):
                if file_name.lower().endswith('.txt'):
                    file_list.append(file_path)
                elif file_name.lower().endswith('.pdf'):
                    file_list.append(file_path)
        return file_list
    
                    
    def text_splitter(self,text: str, max_token_len: int = 600, overlap: int = 150):
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=max_token_len,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
            )
        texts = chunker.create_documents([text])
        # Extract only the page_content
        chunks = [doc.page_content for doc in texts]
        return chunks
        

    
    def read_file_content(self, file_path: str):
        # read according to the file name
        if file_path.endswith('.pdf'):
            content=pdf_to_text(file_path)
            clean_content=remove_ref(content)
            return clean_content
        elif file_path.endswith('.txt'):
            content=load_txt(file_path)
            clean_content=remove_ref(content)
            return clean_content
        else:
            raise ValueError("Unsupported file type")
    
    def get_content(self, max_token_len: int = 600, overlap_content: int = 150):
        '''
        Read the content of the files in the folder and chunk them into smaller pieces.
        Args:
            max_token_len (int): The maximum length of the chunk.
            overlap_content (int): The overlap between the chunks.
        Returns:
            List[str]: A list of chunks.
        '''
        docs = []
        # read content of the file
        for file in self.read_files_list():
            content = self.read_file_content(file)
            chunk_content = self.text_splitter(
                content, max_token_len=max_token_len, overlap=overlap_content)
            docs.extend(chunk_content)
        return docs
                    
                    
        
        
'''    def get_chunk(self, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # Assumed that the text is split by lines

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(self.enc.encode(line))
            if line_len > max_token_len:
                # chunk to multi-rows if one row exceeds max_token_len
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # Avoid splitting across words
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                # Process the last chunk
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk)
                
            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:]+line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text'''
                    
            
'''    def get_chunks_word_aware(self,text: str, max_token_len: int = 600, overlap: int = 150):
        token_len = max_token_len - overlap  # Effective token length for non-overlap chunk
        text=text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
        text=text.strip()
        words = re.findall(r'\S+|\n', text)  # Split by words or newline
        chunks = []
        curr_chunk = []
        curr_len = 0

        for word in words:
            word_len = len(self.enc.encode(word))  # Get the token length of the word
            
            # Check if adding the word exceeds the token limit
            if curr_len + word_len > token_len:
                # Append the current chunk as a complete chunk
                chunks.append(' '.join(curr_chunk))
                
                # Reset the current chunk with overlap words
                curr_chunk = curr_chunk[-overlap:] if overlap > 0 else []
                curr_len = sum(len(self.enc.encode(w)) for w in curr_chunk)  # Recalculate current length

            # Add the current word to the chunk
            curr_chunk.append(word)
            curr_len += word_len

        # Add the last chunk
        if curr_chunk:
            chunks.append(' '.join(curr_chunk))

        return chunks'''