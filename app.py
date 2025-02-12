import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css  # Optional: your custom CSS

# Set your OpenAI API key from Streamlit secrets.
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

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
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore, model_name):
    """Initialize a conversation chain with retrieval memory."""
    llm = ChatOpenAI(model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def message_to_dict(msg):
    """
    Convert a message object to a dictionary.
    If it's already a dict, return it; otherwise, infer the role.
    """
    if isinstance(msg, dict):
        return msg
    class_name = msg.__class__.__name__
    if class_name == "HumanMessage":
        role = "user"
    elif class_name == "AIMessage":
        role = "assistant"
    else:
        role = "unknown"
    return {"role": role, "content": msg.content}

# ------------------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Chat with your assistant", page_icon=":robot:")

    # Inject custom CSS to reserve bottom space and fix the chat input area.
    st.markdown("""
    <style>
      /* Increase bottom padding for the main container */
      .reportview-container .main .block-container {
          padding-bottom: 140px;
      }
      /* Fixed footer styling */
      .fixed-footer {
          position: fixed;
          left: 0;
          bottom: 0;
          width: 100%;
          background-color: #f8f9fa;
          padding: 10px;
          border-top: 1px solid #ddd;
          z-index: 100;
      }
    </style>
    """, unsafe_allow_html=True)

    # Inject JavaScript to scroll to the bottom on page load.
    st.markdown("""
    <script>
      window.onload = function() {
        window.scrollTo(0, document.body.scrollHeight);
      }
    </script>
    """, unsafe_allow_html=True)

    # Optionally include external CSS.
    st.markdown(css, unsafe_allow_html=True)

    # Initialize session state variables.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List of {"role": "user"/"assistant", "content": "..."}
    if "chat_history_archive" not in st.session_state:
        st.session_state.chat_history_archive = []

    # ------------------------------------------------------------------------------
    # Sidebar: Model Selection, PDF Upload, and Chat History Archive
    # ------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Setup")
        # Define available model options.
        model_options = {
            "GPT-4": "gpt-4",
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
            st.session_state.chat_history_archive = []
            st.rerun()
        if st.session_state.chat_history_archive:
            st.subheader("Chat History Archive")
            for idx, conv in enumerate(st.session_state.chat_history_archive):
                with st.expander(f"Conversation {idx + 1}"):
                    for msg in conv:
                        msg_dict = message_to_dict(msg)
                        st.markdown(f"**{msg_dict['role'].capitalize()}:** {msg_dict['content']}")

    # ------------------------------------------------------------------------------
    # Main Chat Interface: Display Messages
    # ------------------------------------------------------------------------------
    st.title("Chat with your assistant")
    if hasattr(st, "chat_message"):
        for msg in st.session_state.messages:
            msg_dict = message_to_dict(msg)
            with st.chat_message(msg_dict["role"]):
                st.markdown(msg_dict["content"])
    else:
        for msg in st.session_state.messages:
            msg_dict = message_to_dict(msg)
            st.markdown(f"**{msg_dict['role'].capitalize()}:** {msg_dict['content']}")

    # ------------------------------------------------------------------------------
    # Fixed Footer: Chat Input Area and Clear Chat Button
    # ------------------------------------------------------------------------------
    st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
    cols = st.columns([4, 1])
    with cols[0]:
        # Use st.chat_input if available; otherwise fallback to st.text_input.
        if hasattr(st, "chat_input"):
            user_input = st.chat_input("Type your message here")
        else:
            user_input = st.text_input("Type your message here", key="fixed_input")
    with cols[1]:
        if st.button("Clear Chat", key="clear_chat_btn"):
            if st.session_state.messages:
                st.session_state.chat_history_archive.append(st.session_state.messages.copy())
            st.session_state.messages = []
            if st.session_state.conversation is not None:
                st.session_state.conversation.memory.clear()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # Process User Input
    # ------------------------------------------------------------------------------
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        if st.session_state.conversation is not None:
            with st.spinner("Assistant is typing..."):
                response = st.session_state.conversation({"question": user_input})
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
        st.rerun()

if __name__ == "__main__":
    main()
