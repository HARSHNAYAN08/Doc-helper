from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Changed from OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from utilities.utils import setup_logger
from typing import IO, Dict, Tuple, List
import pymupdf
import os

Logger = setup_logger(logger_file="app")

# Set Google API key for embeddings
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQqWQEtnl030ru0mvbO9RZegzp3FwGNsI"

def extract_text_with_page_numbers(pdf_file: IO[bytes]) -> Tuple[str, Dict[int, str]]:
    """
    Use PyMuPDF (imported as pymupdf) to extract text from the PDF while keeping track of page numbers.
    """
    pdf_file.seek(0)
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    
    full_text: str = ""
    page_texts: Dict[int, str] = {}
    
    for i, page in enumerate(doc):
        text: str = page.get_text("text")
        full_text += text + "\n"
        page_texts[i + 1] = text
    
    return full_text, page_texts

def process_text_with_splitter(text: str, page_numbers: List[int]) -> FAISS:
    """
    Process text using RecursiveCharacterTextSplitter and create FAISS knowledge base.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    Logger.debug(f"Text split into {len(chunks)} chunks.")
    
    # Use Google Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Use FAISS instead of Chroma
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    Logger.info("Knowledge base created from text chunks.")
    
    # Store the chunks with their corresponding page numbers
    if len(page_numbers) >= len(chunks):
        knowledgeBase.page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
    else:
        knowledgeBase.page_info = {chunk: page_numbers[i % len(page_numbers)] for i, chunk in enumerate(chunks)}
    
    return knowledgeBase
