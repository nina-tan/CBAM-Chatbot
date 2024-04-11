from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import time

PDF_FILES = ["Default transitional.pdf", "Fixes.pdf", "Implementing regulation.pdf", "Importers.pdf", "IT changes.pdf", "NCAs.pdf", "Operators.pdf", "Publications office.pdf", "Q&A.pdf", "User manual.pdf"]

# Custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extract text from pdf
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Adding fallback for pages that return None
    return text

# Convert text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)   
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Use all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Generate conversation chain  
def get_conversationchain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # Use conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vectorstore.as_retriever(),
                            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                            memory=memory)
    return conversation_chain

# Split the response into a generator of words to simulate streaming
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    load_dotenv()
    st.set_page_config(page_title="CBAM Genie", page_icon="💥")
    st.title("CBAM Genie")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="💬"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

    if "conversation_chain" not in st.session_state:
        with st.spinner("Initializing conversation with documents..."):
            current_directory = os.path.dirname(__file__)
            documents_folder = os.path.join(current_directory, "documents")
            pdf_paths = [os.path.join(documents_folder, pdf) for pdf in PDF_FILES]

            raw_text = get_pdf_text(pdf_paths)
            text_chunks = get_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation_chain = get_conversationchain(vectorstore)

    if prompt := st.chat_input("Ask a question about CBAM."):
        # Display user prompt
        with st.chat_message("user", avatar="💬"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        response = st.session_state.conversation_chain({'question': prompt})['answer']

        # Stream the response
        with st.chat_message("assistant", avatar="🤖"):
            full_response = st.write_stream(response_generator(response))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()