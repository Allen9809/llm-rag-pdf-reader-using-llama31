import json
import os
import argparse
from PIL import Image
import streamlit as st  
import time

from groq import Groq 

from functions import load_documents, split_documents, add_to_chroma, initialize_chroma, process_query, clear_chroma_and_file

from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Load config file
working_dic = os.path.dirname(os.path.abspath(__file__))
config_file = json.load(open(f"{working_dic}/config.json"))
groq_api_key = config_file['groq_api_key']
CHROMA_PATH = f"{working_dic}/{config_file['chroma_path']}" 
DATA_PATH = f"{working_dic}/{config_file['data_path']}"

# Save the API key to environment variable
os.environ['GROQ_API_KEY'] = groq_api_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Create the LLM client 
client = Groq()

# Streamlit interface
st.set_page_config(
    page_title="Westie Costco ChatBot",
    page_icon=f"{working_dic}/images/westie.png",
    layout="centered",
)

westie_img = Image.open(f"{working_dic}/images/westie_title.png")

# Initialize the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Stream page title setting
col1, col2 = st.columns([0.4, 1])
with col1:
    st.image(westie_img, width=180)  
with col2:
    st.markdown("# Westie Costco ChatBot")
    st.markdown("### 🥺 Please chat with me~")

# Toggle button to switch between RAG query mode and general chat mode
use_pdf_query_mode = st.checkbox("RAG Mode")

# Buttons for clearing chat history
st.button("Clear Chat History", on_click=st.session_state.chat_history.clear)

# Initialize session state for uploaded files and chat history
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if use_pdf_query_mode:

    # File upload handler for RAG mode
    user_uploaded_files = st.file_uploader("📁 You can also upload PDF files to ask questions about them", type="pdf", accept_multiple_files=True)

    
    # Display chat history
    for message in st.session_state.chat_history:
        avatar_path = f"{working_dic}/images/confused_dog.gif" if message["role"] == "user" else f"{working_dic}/images/westie.png"
        st.chat_message(message["role"], avatar=avatar_path).markdown(message["content"])

    if user_uploaded_files:
        new_files = [file for file in user_uploaded_files if file.name not in st.session_state.uploaded_files]

        if new_files:
            for user_uploaded_file in new_files:
                file_name = user_uploaded_file.name
                file_path = os.path.join(DATA_PATH, file_name)
                with open(file_path, "wb") as f:
                    f.write(user_uploaded_file.getbuffer())
                
                # Create/recreate the vector data store using Chroma
                documents = load_documents()
                chunks = split_documents(documents)
                add_to_chroma(chunks)
                
                st.session_state.uploaded_files.append(file_name)
                
                # Notify user
                message_container = st.empty()
                message_container.success(f"✅ PDF '{file_name}' has been successfully uploaded! \n\n\n\n You can now ask questions about it!!")
                time.sleep(2)
                message_container.empty()

                
                        
    user_prompt = st.chat_input("Ask a question about the uploaded PDF:")
    if user_prompt:
        try:
            st.chat_message("user", avatar=f"{working_dic}/images/confused_dog.gif").markdown(user_prompt)
            # Add user prompt to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Initialize Chroma and get results
            persistent_client, collection = initialize_chroma()
            db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
            results = db.similarity_search_with_score(user_prompt, k=6)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            
            PROMPT_TEMPLATE = """
            Answer the question based only on the following context:

            {context}

            ---

            Answer the question based on the above context: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=user_prompt)
            
            # Create full message history
            messages = [{"role": "user", "content": prompt}] + st.session_state.chat_history
            response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
            rob_response = response.choices[0].message.content

            st.chat_message("assistant", avatar=f"{working_dic}/images/westie.png").markdown(rob_response)
            st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
                        
        except Exception as e:
            st.error(f"❌ Error processing the query: {str(e)}")
            
    if st.button("Clear Uploaded Data"):
        try:    
            user_uploaded_files = []
            st.session_state.uploaded_files.clear()
            clear_chroma_and_file()
            
            # Notify user
            message_container = st.empty()
            message_container.success(f"✅ Database Cleared Successfully!")
            time.sleep(2)
            message_container.empty()     
            
        except Exception as e:  
            st.error(f"❌ The database is empty")
            

else:
    
    # Display chat history
    for message in st.session_state.chat_history:
        avatar_path = f"{working_dic}/images/confused_dog.gif" if message["role"] == "user" else f"{working_dic}/images/westie.png"
        st.chat_message(message["role"], avatar=avatar_path).markdown(message["content"])

        
    # General chat mode
    user_prompt = st.chat_input("Type your question for Costco...")
    if user_prompt:
        
        st.chat_message("user", avatar = f"{working_dic}/images/confused_dog.gif").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # Handle general chat input
        messages = [{"role": "user", "content": user_prompt}] + st.session_state.chat_history
        response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
        rob_response = response.choices[0].message.content
        
        # Display the response   
        st.chat_message("assistant", avatar = f"{working_dic}/images/confused_dog.gif").markdown(rob_response)
        st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
        
