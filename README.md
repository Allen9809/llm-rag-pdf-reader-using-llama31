# Westie Costco ChatBot

The **Westie Costco ChatBot** is an interactive application designed to enhance user engagement through a conversational interface. It enables users to ask questions about Costco products and uploaded PDF documents, leveraging advanced natural language processing capabilities provided by LangChain and Groq. The application supports a Retrieval-Augmented Generation (RAG) mode, allowing users to upload PDF files and query their contents. It maintains a conversational history for an improved user experience and dynamically processes documents to extract relevant information in response to inquiries.

## Features
- **Conversational Interface**: Engage in real-time dialogue with the chatbot.
- **PDF Upload**: Upload PDF documents to ask context-specific questions.
- **Chat History**: Access previous interactions to enhance context and continuity.
- **Clear Chat History**: Easily reset the chat for new interactions.

## Requirements
This application requires Python 3.x, along with the following libraries:
- Streamlit
- LangChain
- Groq
- Pillow

All necessary libraries are specified in `requirements.txt`.

## Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Install Required Packages**:
pip install -r requirements.txt

3. **Create the Configuration File**:
{
    "groq_api_key": "YOUR_GROQ_API_KEY",
    "chroma_path": "path/to/chroma",
    "data_path": "path/to/data"
}
