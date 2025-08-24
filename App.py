import streamlit as st
from typing import Dict, Any, Optional, List
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI  # Changed from langchain_openai
from langchain_community.callbacks.manager import get_openai_callback
from modules.process_data import extract_text_with_page_numbers, process_text_with_splitter
from utilities.utils import setup_logger
import os

# Setup logger
Logger = setup_logger(logger_file="app")

# Set Google API key environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQqWQEtnl030ru0mvbO9RZegzp3FwGNsI"

def load_css(file_path: str) -> None:
    """Load custom CSS from file."""
    try:
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_path}")
    except Exception as e:
        st.error(f"Error loading CSS: {str(e)}")

def main() -> None:
    """Main application function."""
    # Load external CSS
    load_css("utilities/styles/main.css")
    
    # Header
    st.markdown(
        '<div class="main-header">Legal Document Q&A Assistant</div>',
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your legal document (PDF)", type=["pdf"])
    
    # Question input
    user_question = st.text_input("Ask a question about the document:")
    
    # Initialize Gemini LLM with native Google integration
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key="AIzaSyBQqWQEtnl030ru0mvbO9RZegzp3FwGNsI"
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        st.stop()
    
    # Process document button
    if st.button("Process Document"):
        # Validation checks
        if uploaded_file is None:
            st.error("Please upload a PDF document first.")
            return
            
        if not user_question.strip():
            st.error("Please enter a question about the document.")
            return
        
        # Show processing indicator
        with st.spinner("Processing document and generating answer..."):
            try:
                # Extract text from PDF
                text_data, page_numbers = extract_text_with_page_numbers(uploaded_file)
                
                if not text_data:
                    st.error("Could not extract text from the PDF.")
                    return
                
                # Process text with splitter
                docs = process_text_with_splitter(text_data, list(page_numbers.keys()))
                
                if not docs:
                    st.error("Could not process the document text.")
                    return
                
                # Load QA chain
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Generate answer
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs.similarity_search(user_question, k=5), question=user_question)
                    
                    # Display results
                    st.success("Answer generated successfully!")
                    
                    with st.expander("📄 **Answer**", expanded=True):
                        st.write(response)
                    
                    st.info(f"✅ Document processed successfully")
                        
            except Exception as e:
                st.error(f"❌ An error occurred while processing: {str(e)}")
                Logger.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
