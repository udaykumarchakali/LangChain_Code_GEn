
import os
import tempfile
import gradio as gr

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

message = """
Answer this question using the provided context only.
If the information is not available in the context,
just reply with "I don't know".

{input}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [("human", message)]
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# this is function create for file loader
def load_pdf_into_vectorstore(file):
    
    if file is None:
        return None, "Please upload a PDF file."
    #  this store the file path means location 
    file_path = file

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chromadb11"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    chain = {
        "input": RunnablePassthrough(),
        "context": retriever
    } | prompt | llm

    return chain, "Document uploaded successfully. You can chat now."

def chat(message, history, qa_chain):
    # if user not uploaded the file so chain also not there in that situation return the empty history 
    if qa_chain is None:
        return history or [],""

    response = qa_chain.invoke(message)

    history = history or []

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.content}
    ]

    return history,""


with gr.Blocks() as demo:   
    qa_chain_state = gr.State(None)

    with gr.Row():

        with gr.Column():

            file = gr.File(
                label="Upload your PDF file",
                file_types=[".pdf"],
                type="filepath"
            )

            upload = gr.Button(
                value="Upload",
                variant="primary"
            )

        label = gr.Textbox()

    chatbot = gr.Chatbot(
        label="Talk to the Document"
    )

    msg = gr.Textbox()

    upload.click(
        fn=load_pdf_into_vectorstore,
        inputs=[file],
        outputs=[qa_chain_state,label]
    )

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot,qa_chain_state],
        outputs=[chatbot,msg]
    )


demo.launch(debug=True)
