import streamlit as st
from  PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
def get_pdf_text(pdf_file):
    """
    This function takes a list of PDF files and returns 
    a string containing the text of all the PDFs.

    :param pdf_file: List of PDF files
    :return: String containing the text of all the PDFs
    """

#    text = ""
#    for pdf in pdf_file:
#        try:
#            pdf_reader = PdfReader(pdf)
#            for page in pdf_reader.pages:
#                text += page.extract_text()
#        except Exception as e:
#            st.error(f"Error reading PDF: {str(e)}")
#   
    text = ""
    for pdf in pdf_file:
        pdf_file_like = BytesIO(pdf)
        pdf_bytes = pdf.read(pdf_file_like)
        
        pdf_reader = PdfReader(pdf_file_like)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text

def get_text_chunks(text):
    """
    This function takes a string of text and returns 
    a list of strings, each of which is a chunk of 1000 words.

    :param text: String of text
    :return: List of strings
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(chunks):
    """
    This function takes a list of strings and creates a 
    FAISS vectorstore.

    :param chunks: List of strings
    :return: FAISS vectorstore
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """Answer the question as truthfully as possible 
                        and in as much detail as possible using the 
                        context below, make sure to provide all the details,
                        if you don't know the answer, just say "I don't know"
                        don't try to make up an answer.\n\n

                        Context: {context}\n
                        Question: {question}\n
                        Answer:
                        
                    """
                    
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"], output_input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain
    
def user_input(use_question):
    """
    This function takes a user's question and asks the chain of questions 
    created by the Gemini-Pro model to answer the question based on the 
    context of the documents in the database.
    Parameters
    ----------
    use_question : str
        The user's question
    Returns
    -------
    str
        The answer to the question
    Notes
    -----
    This function uses the Gemini-Pro model to generate a chain of questions
    based on the context of the documents in the database. The function then
    asks the chain of questions to answer the user's question. The answer is
    then returned to the user.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(use_question)
    
    chain = get_conversation_chain()
    
    response = chain(
        {"input_documents": docs, 
        "question": use_question},
        return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])
    
def main():
    st.set_page_config(page_title="TalkingPDFs", page_icon=":books:")
    st.header("Talk to the PDFs :books:")
    
    user_question = st.text_input("Ask a Question from the PDF files:")
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",
                                    type="pdf",
                                    accept_multiple_files=True)
        if pdf_docs is not None:
            st.button("Process")
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vectorstore(text_chunks)
                st.success("Done!")
                st.write("Your PDFs have been processed. You can now ask questions about them.")
                st.write("If you want to upload more PDFs, please do so in the sidebar.")
                
if __name__ == '__main__':
    main()