import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, model_name):
    llm = ChatOpenAI(model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="Chat with PDF :books:", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialise session state variables if not already set
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_history_archive" not in st.session_state:
        st.session_state.chat_history_archive = []

    st.header("Chat with PDF :books:")

    # Text input for the question
    user_question = st.text_input("Ask a question about your documents:")

    # Clear Chat button placed right next to the input.
    if st.button("Clear Chat"):
        # Archive the current conversation (if any)
        if st.session_state.chat_history:
            st.session_state.chat_history_archive.append(st.session_state.chat_history)
        st.session_state.chat_history = []
        if st.session_state.conversation is not None:
            st.session_state.conversation.memory.chat_history = []
        st.experimental_rerun()  # Rerun to update the UI

    # Container for chat messages (displayed just below the input)
    chat_container = st.container()

    # If a new question is submitted, get a response and display the conversation
    if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(user_template.replace("{{MSG}}", message.content),
                                unsafe_allow_html=True)
                else:
                    st.markdown(bot_template.replace("{{MSG}}", message.content),
                                unsafe_allow_html=True)

    # Sidebar for model selection, file uploading, and chat history archive
    with st.sidebar:
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-4-o": "gpt-4",        # Adjust if you have different settings for '4o'
            "GPT-4-mini": "gpt-4-mini",  # Note: ensure this model is available as intended
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        model_choice = st.selectbox("Select GPT Model", list(model_options.keys()))
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Add Data'",
            accept_multiple_files=True
        )
        if st.button("Add Data"):
            with st.spinner("Adding Data..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split the text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create the vector store
                vectorstore = get_vectorstore(text_chunks)

                # Initialise the conversation chain using the selected GPT model
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, model_options[model_choice]
                )
                st.experimental_rerun()

        # Display the archived chat history in the sidebar
        if st.session_state.chat_history_archive:
            st.subheader("Chat History Archive")
            for idx, conv in enumerate(st.session_state.chat_history_archive):
                with st.expander(f"Conversation {idx + 1}"):
                    for j, message in enumerate(conv):
                        if j % 2 == 0:
                            st.markdown(user_template.replace("{{MSG}}", message.content),
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(bot_template.replace("{{MSG}}", message.content),
                                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
