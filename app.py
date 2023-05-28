from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Magic")
    st.header("PDF Magic")

    #upload
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    #extract
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #split 
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1200,
            #purpose is to give it more context
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text)
        
        # st.write(chunks)
        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("What do you want to know about from your PDF?")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            #chose a llm
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback as cb:
                response = chain.run(input_documents=docs, question=user_question)
                #track api token usage/spending
                print(cb)
            
            st.write(response)

    # This is for READING CSV FILE
    # Load the OpenAI API key from the environment variable
    # if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    #     print("OPENAI_API_KEY is not set")
    #     exit(1)
    # else:
    #     print("OPENAI_API_KEY is set")

    # st.set_page_config(page_title="Ask your CSV")
    # st.header("Ask your CSV 📈")

    # csv_file = st.file_uploader("Upload a CSV file", type="csv")
    # if csv_file is not None:

    #     agent = create_csv_agent(
    #         OpenAI(temperature=0), csv_file, verbose=True)

    #     user_question = st.text_input("Ask a question about your CSV: ")

    #     if user_question is not None and user_question != "":
    #         with st.spinner(text="In progress..."):
    #             st.write(agent.run(user_question))

if __name__ == '__main__':
    main()