import os
from dotenv import load_dotenv
import ollama
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain.llms import Ollama
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def get_vector_store(text_chunks):
    try:
        # Load the SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for each chunk
        embeddings = [model.encode(chunk, convert_to_tensor=False) for chunk in text_chunks]

        # Convert the embeddings to the format required by FAISS
        embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # Create a FAISS vector store
        vector_store = FAISS.from_texts(text_chunks, embedding_function)

        # Save the vector store locally
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")

# def get_vector_store(text_chunks):
#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Force using CPU
        
#         def embed_fn(texts):
#             return model.encode(texts, convert_to_tensor=False).tolist()

#         vector_store = FAISS.from_texts(text_chunks, embedding=embed_fn)
#         vector_store.save_local("faiss_index")
#         st.success("Vector store created successfully!")
#     except Exception as e:
#         st.error(f"Failed to create vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "Answer is not available in the context." 
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = Ollama(model="llama3.2")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

    return chain
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = load_qa_chain(ollama.chat(model="llama3.2"), chain_type="stuff", prompt=prompt)
    # return chain

def user_input(user_question):
    try:
        # Load the embedding model
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        # Load the FAISS vector store with the dangerous deserialization flag
        new_db = FAISS.load_local("faiss_index", embeddings=embedding,allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = new_db.similarity_search(user_question)

        # Get the conversational chain
        chain = get_conversational_chain()
        print(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF Chatbot 🤖")

    with st.sidebar:
        st.title("📁 PDF Upload Section")
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")

    user_question = st.text_input("Ask a Question:")
    if user_question:
        user_input(user_question)

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
