import json
import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from chromadb import PersistentClient

# Load the configuration file containing paths and API keys
working_dic = os.path.dirname(os.path.abspath(__file__))
config_file = json.load(open(f"{working_dic}/config.json"))
groq_api_key = config_file['groq_api_key']
CHROMA_PATH = f"{working_dic}/{config_file['chroma_path']}" 
DATA_PATH = f"{working_dic}/{config_file['data_path']}"

def load_documents():
    """Load all PDF documents from the specified directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    """
    Split the documents into smaller chunks using a recursive character-based text splitter.
    This helps in processing documents more efficiently.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each chunk of the document.
    IDs are based on the document's page number and chunk index.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the current page is the same as the last, increment the chunk index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign a unique ID to the chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add the chunk ID to the metadata
        chunk.metadata["id"] = chunk_id
        
    return chunks

def add_to_chroma(chunks):
    """
    Add or update document chunks in the Chroma vector database.
    Only new chunks are added to avoid duplication.
    """
    # Initialize Chroma persistent client and collection
    persistent_client, collection = initialize_chroma()
    db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())

    # Calculate unique chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing document IDs from the database
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out chunks that already exist in the database
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def clear_chroma_and_file():
    """
    Clear all files in the data directory and reset the Chroma database.
    This is useful for a complete refresh of the document and vector data.
    """
    # Delete all files in the data path
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        
    # Reset Chroma vector database
    persistent_client, collection = initialize_chroma()
    db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
    db.delete(ids=db.get()['ids'])
    print(f"👉 Number of existing documents in DB after deleting: {len(set(db.get()['ids']))}")

def initialize_chroma():
    """
    Initialize the Chroma persistent client and create or retrieve the collection.
    Returns the client and collection objects.
    """
    persistent_client = PersistentClient(path=CHROMA_PATH)
    collection = persistent_client.get_or_create_collection("vecdb")
    return persistent_client, collection

def process_query(query_text):
    """
    Process the user's query using the Chroma database and a language model (LLM).
    Retrieves the most relevant document chunks and formulates a response.
    """
    try:
        # Initialize Chroma client and collection
        persistent_client, collection = initialize_chroma()
        db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
        
        # Perform similarity search to retrieve relevant document chunks
        results = db.similarity_search_with_score(query_text, k=6)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Define the prompt template for the LLM
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}-

        ---

        Answer the question based on the above context: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Display the user's query in the chat UI
        st.chat_message("user", avatar=f"{working_dic}/images/confused_dog.gif").markdown(query_text)
        st.session_state.chat_history.append({"role": "user", "content": query_text})
        
        # Construct the message history for the LLM
        messages = [{"role": "assistant", "content": "answer the question briefly and efficiently!!"}, *st.session_state.chat_history]
        print(messages)
        
        # Get the response from the LLM model
        response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
        
        # Extract and display the assistant's response
        rob_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
        
        with st.chat_message("assistant", avatar=f"{working_dic}/images/westie.png"):
            st.markdown(rob_response)
        
    except Exception as e:
        st.error(f"❌ Error processing the query: {str(e)}")
