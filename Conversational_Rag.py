from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever

load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini")

# load the pdf
loader=PyPDFLoader(file_path="Hosipital.pdf")
documents=loader.load()
print(len(documents))

# split documents into chunks
text_splitter=CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30,
    separator="\n"
)
splits = text_splitter.split_documents(
    documents=documents
)
print(len(splits))

# create embeddings
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vector database

vectorstore=FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever=vectorstore.as_retriever()

# create chat history
chat_history=[]

# create question-answer prompt
messages="""
Answer this question using the provided context only.
If you don't know the answer, just say "I don't know".
{input}
Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", messages)
    ]
)

# create question answer chain

question_answer_chain=create_stuff_documents_chain(
    llm,
    prompt
)

# create normal RAG Chain

rag_chain=create_retrieval_chain(
    retriever,
    question_answer_chain
)

# First user question
input="""
give me Medical Departments,

Give me only names as list"""

# invoke normal Rag
response=rag_chain.invoke(
    {
        "chat_history":chat_history,
        "input":input
    }
)

# print Answer
print(response["answer"])

# store chat history
chat_history.extend(
    [
        HumanMessage(content=input),
        AIMessage(content=response["answer"])
    ]
)

# create contextualize quesry prompt
contextualize_q_system_prompt="""
Give above chat history and the below latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history.

Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.

Below is the latest question:
{input}
"""

# create contextualize prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# create history aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# test history aware retriever
result = history_aware_retriever.invoke(
    {
        "chat_history": chat_history,
        "input": """
        give me Medical Departments,

Give me only names as list
        """
    }
)

print(result)

# create final system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "

    "Answer this question using the provided context only. "
    "If you don't know the answer, just say 'I don't know'"

    "\n\n"

    "{context}"
)

# create question answer prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# create question answer chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

# Create conversational RAG chain
contextual_rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

# create empty chat history
chat_history = []

# Ask first question
question = """
give me Medical Departments,

Give me only names as list
"""

# Invoke conversational RAG
ai_msg_1 = contextual_rag_chain.invoke(
    {
        "input": question,
        "chat_history": chat_history
    }
)



# print first answer
print(ai_msg_1["answer"])


#Store first Converstation
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"])
    ]
)


#Ask Follow-up question
question_2 = """
give me Medical Departments,

Give me only names as list and lines
"""


# invoke second question

ai_msg_2 = contextual_rag_chain.invoke(
    {
        "input": question_2,
        "chat_history": chat_history
    }
)


#print final answer
print(ai_msg_2["answer"])


# save second conversation
chat_history.extend(
    [
        HumanMessage(content=question_2),
        AIMessage(content=ai_msg_2["answer"])
    ]
)


