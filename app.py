import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css  # Make sure you have this file with your CSS styles

# Set your OpenAI API key from Streamlit secrets.
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore, model_name):
    """Initialise a conversation chain with retrieval memory."""
    llm = ChatOpenAI(model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def main():
    # Set up the page title and inject CSS.
    st.set_page_config(page_title="Chat with your assistant", page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)

    # Initialise session state variables.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        # messages will be a list of dictionaries: {"role": "user"/"assistant", "content": "…"}
        st.session_state.messages = []

    # ─── SIDEBAR: Setup, File Upload, and Chat History Clear ─────────────────────────
    with st.sidebar:
        st.header("Setup")
        model_options = {
            "GPT-4": "gpt-4",
            "GPT-4-o": "gpt-4",
            "GPT-4-mini": "gpt-4-mini",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        model_choice = st.selectbox("Select GPT Model", list(model_options.keys()))
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Add Data"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore, model_options[model_choice]
                    )
                    st.success("Data added successfully!")
                    st.rerun()
            else:
                st.warning("Please upload at least one PDF.")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # ─── MAIN CHAT INTERFACE ─────────────────────────────────────────────────────────
    st.title("Chat with your assistant")

    # Display previous conversation messages.
    # If available, use the new st.chat_message component for a ChatGPT-like UI.
    if hasattr(st, "chat_message"):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        # Fallback if the new chat components are not available.
        for msg in st.session_state.messages:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    # Chat input area.
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("Type your message here")
    else:
        user_input = st.text_input("Type your message here")
    
    if user_input:
        # Append the user's message.
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Check if the conversation chain has been set up.
        if st.session_state.conversation is not None:
            with st.spinner("Assistant is typing..."):
                # Call the conversation chain with the new question.
                response = st.session_state.conversation({"question": user_input})
                # Retrieve the latest assistant response.
                if response.get("chat_history"):
                    latest_message = response["chat_history"][-1]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": latest_message.content
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Sorry, I couldn't generate a response."
                    })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please upload your documents and click 'Add Data' to start."
            })
        st.rerun()  # Refresh the UI to display the new messages.

if __name__ == "__main__":
    main()
