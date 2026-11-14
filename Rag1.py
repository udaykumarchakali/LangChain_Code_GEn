import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)

load_dotenv()

# load PDF
loader=PyPDFLoader("Hosipital.pdf")
documents=loader.load()

# len of documents 
print("Total pages:",len(documents))

# split documents into chunks
text_splitter= CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)

docs= text_splitter.split_documents(documents)
 
print("Total chunks:",len(docs))

# create embedding model
embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# create FAISS vectore store
vectorstore=FAISS.from_documents(
    docs,embeddings
)

# Save FAISS vector store locally
vectorstore.save_local("faiss_store1")

# load existing FAISS vector store 
vectorstore=FAISS.load_local(
    "faiss_store1",
    embeddings,
    allow_dangerous_deserialization=True
)

# create retriever
retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

message="""
        Answer this question using the provided context only.
        If the information is not available in the context , just reply with "I don't Know
        {input}
        Context:
        {context}
        """
prompt =ChatPromptTemplate.from_messages(
    [("human",message)],
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
chain={"input":RunnablePassthrough(),"context":retriever} | prompt | llm
response=chain.invoke("give me introduction")
print(response.content)





   