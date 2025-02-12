import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css
import os

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Function to create conversation chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatOpenAI(model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    chat_container = """<div style="display: flex; flex-direction: column; gap: 10px;">"""

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            chat_container += f"""
            <div style="display: flex; justify-content: flex-end;">
                <div style="background-color: #dcf8c6; padding: 8px 12px; border-radius: 10px; max-width: 60%;">{message.content}</div>
            </div>
            """
        else:
            chat_container += f"""
            <div style="display: flex; justify-content: flex-start;">
                <div style="background-color: #f1f0f0; padding: 8px 12px; border-radius: 10px; max-width: 60%;">{message.content}</div>
            </div>
            """
    chat_container += "</div>"
    st.markdown(chat_container, unsafe_allow_html=True)


# Main function
def main():
    st.set_page_config(page_title="Chat with PDF :books:", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-4-o": "gpt-4",
            "GPT-4-mini": "gpt-4-mini",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        model_choice = st.selectbox("Select GPT Model", list(model_options.keys()))
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Add Data'", accept_multiple_files=True)
        if st.button("Add Data"):
            with st.spinner("Adding Data..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, model_options[model_choice])

if __name__ == "__main__":
    main()
