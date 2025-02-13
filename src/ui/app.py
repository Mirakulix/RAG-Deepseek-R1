import streamlit as st
import requests
import json
import os
from typing import Optional
import pandas as pd

# Configure the API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)

def query_documents(query: str, context_size: int = 3) -> Optional[dict]:
    """Query the RAG system API."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"text": query, "context_size": context_size}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error querying API: {str(e)}")
        return None

def upload_document(content: str, metadata: dict) -> bool:
    """Upload a document to the RAG system."""
    try:
        response = requests.post(
            f"{API_URL}/documents",
            json={"content": content, "metadata": metadata}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return False

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "md"])
    
    if uploaded_file is not None:
        content = uploaded_file.read()
        
        if uploaded_file.type == "application/pdf":
            # Handle PDF
            # You'll need to add PDF processing logic here
            pass
        else:
            # Handle text files
            content = content.decode("utf-8")
            
        metadata = {
            "filename": uploaded_file.name,
            "type": uploaded_file.type,
            "size": uploaded_file.size
        }
        
        if st.button("Upload Document"):
            if upload_document(content, metadata):
                st.success("Document uploaded successfully!")
            else:
                st.error("Failed to upload document")

# Main query interface
st.title("RAG System Query Interface")

query = st.text_area("Enter your query:", height=100)
context_size = st.slider("Context Size", min_value=1, max_value=10, value=3)

if st.button("Submit Query"):
    if query:
        with st.spinner("Processing query..."):
            result = query_documents(query, context_size)
            
            if result:
                st.subheader("Response")
                st.write(result["response"])
                
                # Show relevant documents if available
                if "context" in result:
                    with st.expander("View Related Documents"):
                        for idx, doc in enumerate(result["context"], 1):
                            st.markdown(f"**Document {idx}**")
                            st.text(doc)
    else:
        st.warning("Please enter a query")