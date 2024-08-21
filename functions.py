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

# load config file
working_dic = os.path.dirname(os.path.abspath(__file__))
config_file = json.load(open(f"{working_dic}/config.json"))
groq_api_key = config_file['groq_api_key']
CHROMA_PATH = f"{working_dic}/{config_file['chroma_path']}" 
DATA_PATH = f"{working_dic}/{config_file['data_path']}"


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/xxx.pdf:6:2" --- Page Number : Chunk Index
    
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
        
    return chunks

def add_to_chroma(chunks):

    # Initialize PersistentClient with the correct path
    persistent_client, collection = initialize_chroma()

    db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def clear_chroma_and_file():
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        
    persistent_client, collection = initialize_chroma()
    db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
    db.delete(ids = db.get()['ids'])
    print(f"👉 Number of existing documents in DB after deleting: {len(set(db.get()['ids']))}")
        
        
def initialize_chroma():
    """Initialize the Chroma database and return the client and collection."""
    persistent_client = PersistentClient(path=CHROMA_PATH)
    collection = persistent_client.get_or_create_collection("vecdb")
    return persistent_client, collection

def process_query(query_text, ):
    """Process the user query using Chroma and LLM model."""
    try:
        persistent_client, collection = initialize_chroma()
        db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
        
        results = db.similarity_search_with_score(query_text, k=6)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}-

        ---

        Answer the question based on the above context: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        st.chat_message("user", avatar=f"{working_dic}/images/confused_dog.gif").markdown(query_text)
        st.session_state.chat_history.append({"role": "user", "content": query_text})
        
        # Prompt the LLM model using the user's message
        messages = [{"role": "assistant", "content": "answer the question briefly and efficiently!!"}, *st.session_state.chat_history]
        print(messages)
        response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
        
        rob_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
        
        # Display the response
        with st.chat_message("assistant", avatar=f"{working_dic}/images/westie.png"):
            st.markdown(rob_response)
        
    except Exception as e:
        st.error(f"❌ Error processing the query: {str(e)}")
        
        
