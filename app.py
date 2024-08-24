__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


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
    layout="centered")
westie_img = Image.open(f"{working_dic}/images/westie_title.png")

# Stream page title setting
col1, col2 = st.columns([0.4, 1])
with col1:
    st.image(westie_img, width=180)  
with col2:
    st.markdown("# Westie Costco ChatBot")
    st.markdown("### 🥺 Please chat with me~")



# Initialize session state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize the chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the file uploader key
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
# Toggle button to switch between RAG query mode and general chat mode
use_pdf_query_mode = st.checkbox("PDF Reading Mode")    



# Define the stype that the chatbot talks
BACKGROUND = "personality: funny,naughty,helpful,kind, efficient; \n species: 3 years old westie dog named Costco; \n hobby: sleep under coach, strech, eat"

# PDF reading mode set-up
if use_pdf_query_mode:

  # Button for clearing the database and chat history
    if st.button("Clear Uploaded Data"):
                
        # Clear uploaded files and session state
        st.session_state["file_uploader_key"] += 1
        st.session_state.uploaded_files.clear()
        clear_chroma_and_file()  

        # Clear chat history
        st.session_state.chat_history.clear()

        # Notify user of success
        message_container = st.empty()
        message_container.success("✅ Database and Chat History Cleared Successfully!")
        time.sleep(2)  
        message_container.empty()
        st.rerun()

    # File upload handler 
    user_uploaded_files = st.file_uploader("📁 You can upload PDF files and ask realated questions", 
                                           type="pdf", 
                                           accept_multiple_files=True,
                                           key= st.session_state["file_uploader_key"])
    
    # # Display chat history
    for message in st.session_state.chat_history[1:]:
        avatar_path = f"{working_dic}/images/confused_dog.gif" if message["role"] == "user" else f"{working_dic}/images/westie.png"
        st.chat_message(message["role"], avatar=avatar_path).markdown(message["content"])
        
    
    if user_uploaded_files:
    
        # Ensure the DATA_PATH directory exists
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            
        # only write new files into the data folder and DB 
        new_files = [file for file in user_uploaded_files if file.name not in st.session_state.uploaded_files]

        if new_files:
            for user_uploaded_file in new_files:
                file_name = user_uploaded_file.name
                file_path = os.path.join(DATA_PATH, file_name)
                with open(file_path, "wb") as f:
                    f.write(user_uploaded_file.getbuffer())
                
                # add uploaded file into DB as vectors
                documents = load_documents()
                chunks = split_documents(documents)
                add_to_chroma(chunks)
                st.session_state.uploaded_files.append(file_name)
                # st.session_state.chat_history.append({"role": "user", "content": f"the user uploaded {file_name}"})

                # Notify user of success
                message_container = st.empty()
                message_container.success(f"✅ PDF '{file_name}' has been successfully uploaded! \n\n\n\n You can now ask questions about it!!")
                time.sleep(2)
                message_container.empty()
        
    # User input handler                    
    user_prompt = st.chat_input("Ask a question about the uploaded PDF:")

    if user_prompt:

        try:
            # Display and add user prompt to chat history
            st.chat_message("user", avatar=f"{working_dic}/images/confused_dog.gif").markdown(user_prompt)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Initialize Chroma and get promting results
            persistent_client, collection = initialize_chroma()
            db = Chroma(client=persistent_client, collection_name="vecdb", embedding_function=HuggingFaceEmbeddings())
            results = db.similarity_search_with_score(user_prompt, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            
            PROMPT_TEMPLATE = """
            Firstly, here is the background information of you. Please draft all your response based on this profile:

            {background}

            
            ---

            Secondly, answer the question based only on the following context:

            {context}

            ---

            Lastly, answer the question based on the above context: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(background = BACKGROUND, context=context_text, question=user_prompt)
            
            # Generate the robot response using the whole chat history
            messages = [{"role": "user", "content": prompt}] + st.session_state.chat_history
            response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
            rob_response = response.choices[0].message.content

            # Display and add response to chat history
            st.chat_message("assistant", avatar=f"{working_dic}/images/westie.png").markdown(rob_response)
            st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
                        
        except Exception as e:
            st.error(f"❌ Error processing the query: {str(e)}")



else:
    
    if st.session_state.chat_history == []:
        st.session_state.chat_history.append({"role": "user", "content": f"Here is the background information of you. Please draft all your response based on this profile: {BACKGROUND} "})
        
    # Display chat history
    for message in st.session_state.chat_history[1:]:
        avatar_path = f"{working_dic}/images/confused_dog.gif" if message["role"] == "user" else f"{working_dic}/images/westie.png"
        st.chat_message(message["role"], avatar=avatar_path).markdown(message["content"])

    # General chat mode
    user_prompt = st.chat_input("Type your question for Costco...")
    if user_prompt:

        # Display and add user prompt to chat history
        st.chat_message("user", avatar = f"{working_dic}/images/confused_dog.gif").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Generate the robot response
        messages = [{"role": "user", "content": user_prompt}] + st.session_state.chat_history
        response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
        rob_response = response.choices[0].message.content
        
        # Display and add response to chat history 
        st.chat_message("assistant", avatar = f"{working_dic}/images/westie.png").markdown(rob_response)
        st.session_state.chat_history.append({"role": "assistant", "content": rob_response})
        
