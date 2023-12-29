import os
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from PIL import Image
import pytesseract
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from google.api_core.client_options import ClientOptions
import os
import io
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import vision_v1


import os
import dotenv
openai_api_key = os.getenv("openai_api_key")

# Google Cloud settings
PROJECT_ID = os.getenv("dev-mgmt-cdu-da-ea")
LOCATION = os.getenv("us")
PROCESSOR_ID = os.getenv("ad3ff60a41feb9e5")

    # Set the path to the service account credentials JSON file for Document AI
credential_path = r"C:\Users\91900\OneDrive\Desktop\Gen_AI_Projects\Image\Chat_with_multiple_formats_including_image\Q&A_with_image_aswell.py\dev-mgmt-cdu-da-ea-53c6cfcb9d0c.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


def get_image_text(image_file):
    # Initialize Vision client
    vision_client = vision_v1.ImageAnnotatorClient()

    # Read file content
    image_content = image_file.getvalue()

    # Perform OCR on the image using Vision API
    image = vision_v1.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    extracted_text = response.text_annotations[0].description if response.text_annotations else ""
    return extracted_text

def get_data_text(data_files):
    text = ""
    for data_file in data_files:
        if data_file.type in ["image/jpeg", "image/jpg", "image/png"]:
            text += get_image_text(data_file)
        else:
            st.error(f"File type not supported: {data_file.type}")
            return None
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key,model_name="gpt-4",temperature=0.9)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with multiple data files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple data files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        data_files = st.file_uploader(
            "Images here and click on 'Process'",
            accept_multiple_files=True,
            type=['jpg', 'jpeg', 'png']
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get data text
                raw_text = get_data_text(data_files)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
