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

    # Initialise session state variables if not already set.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_history_archive" not in st.session_state:
        st.session_state.chat_history_archive = []
    if "text_input_key" not in st.session_state:
        st.session_state.text_input_key = 0

    st.header("Chat with PDF :books:")

    # Text input for the user's question with a dynamic key.
    user_question = st.text_input(
        "Ask a question about your documents:",
        key=f"user_question_{st.session_state.text_input_key}"
    )

    # Clear Chat button placed immediately below the text input.
    if st.button("Clear Chat"):
        # Archive the current conversation if there is any.
        if st.session_state.chat_history:
            st.session_state.chat_history_archive.append(st.session_state.chat_history)
        st.session_state.chat_history = []
        if st.session_state.conversation is not None:
            st.session_state.conversation.memory.clear()
        # Increment the key to create a new text input widget.
        st.session_state.text_input_key += 1
        st.rerun()

    # Container for chat messages.
    chat_container = st.container()

    # If a new question is provided, process it and update the conversation history.
    if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

    # Group the conversation history into pairs (user question and bot response)
    conversation_pairs = []
    history = st.session_state.chat_history
    i = 0
    while i < len(history):
        if i + 1 < len(history):
            conversation_pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            # In case there's an unmatched message.
            conversation_pairs.append((history[i], None))
            i += 1

    # Display the conversation pairs in reverse order (latest at the top).
    with chat_container:
        for user_msg, bot_msg in reversed(conversation_pairs):
            st.markdown(user_template.replace("{{MSG}}", user_msg.content),
                        unsafe_allow_html=True)
            if bot_msg is not None:
                st.markdown(bot_template.replace("{{MSG}}", bot_msg.content),
                            unsafe_allow_html=True)

    # Sidebar for model selection, file uploading, and chat history archive.
    with st.sidebar:
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-4-o": "gpt-4",       # Adjust if you have different settings for '4o'
            "GPT-4-mini": "gpt-4-mini", # Note: ensure this model is available as intended
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
                # Extract text from PDFs.
                raw_text = get_pdf_text(pdf_docs)
                # Split the text into chunks.
                text_chunks = get_text_chunks(raw_text)
                # Create the vector store.
                vectorstore = get_vectorstore(text_chunks)
                # Initialise the conversation chain using the selected GPT model.
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, model_options[model_choice]
                )
                st.rerun()

        # Clear Chat History button to clear the archived history.
        if st.button("Clear Chat History"):
            st.session_state.chat_history_archive = []
            st.rerun()

        # Display the archived chat history in the sidebar.
        if st.session_state.chat_history_archive:
            st.subheader("Chat History Archive")
            for idx, conv in enumerate(st.session_state.chat_history_archive):
                with st.expander(f"Conversation {idx + 1}"):
                    # Group each archived conversation into pairs before displaying.
                    archived_pairs = []
                    j = 0
                    while j < len(conv):
                        if j + 1 < len(conv):
                            archived_pairs.append((conv[j], conv[j + 1]))
                            j += 2
                        else:
                            archived_pairs.append((conv[j], None))
                            j += 1
                    for user_msg, bot_msg in archived_pairs:
                        st.markdown(user_template.replace("{{MSG}}", user_msg.content),
                                    unsafe_allow_html=True)
                        if bot_msg is not None:
                            st.markdown(bot_template.replace("{{MSG}}", bot_msg.content),
                                        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
