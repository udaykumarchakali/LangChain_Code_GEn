
import os
import gradio as gr

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)

# Ragas imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    AnswerCorrectness
)

load_dotenv()

# -------------------- PROMPT --------------------

message = """
Answer this question using the provided context only.
If the information is not available in the context,
just reply with "I don't know".

{input}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("human", message)
])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Embeddings
embeddings = OpenAIEmbeddings()


# -------------------- LOAD PDF --------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_pdf_into_vectorstore(file):

    if file is None:
        return None, "Please upload a PDF file."

    file_path = file

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )

    docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chromadb11"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # First retrieve documents and keep the original documents
    retrieval_chain = RunnableParallel({
        "input": RunnablePassthrough(),
        "context": retriever
    })

    # Generate answer using retrieved documents number into textin
    qa_chain = retrieval_chain | RunnablePassthrough.assign(
        answer=(
            RunnableLambda(
                lambda data: {
                    "input": data["input"],
                    "context": format_docs(data["context"])
                }
            )
            | prompt
            | llm
        )
    )

    return qa_chain, "Document uploaded successfully. You can chat now."



# -------------------- CHAT FUNCTION --------------------

def chat(message, history, qa_chain):

    if qa_chain is None:
        return history or [], ""

    result = qa_chain.invoke(message)

    answer = result["answer"].content

    history = history or []

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return history, ""


# -------------------- RAGAS EVALUATION --------------------

def evaluate_rag(qa_chain, question, reference):

    if qa_chain is None:
        return "Please upload a PDF before evaluation."

    if not question or not reference:
        return "Please enter both a test question and a ground-truth answer."

    try:
        # Run the same RAG chain
        result = qa_chain.invoke(question)

        generated_answer = result["answer"].content

        # Extract retrieved context chunks
        retrieved_contexts = [
            doc.page_content for doc in result["context"]
        ]

        # Prepare evaluation data
        evaluation_data = {
            "user_input": [question],
            "retrieved_contexts": [retrieved_contexts],
            "response": [generated_answer],
            "reference": [reference]
        }

        dataset = Dataset.from_dict(evaluation_data)

        # Ragas evaluation metrics
        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            AnswerCorrectness()
        ]

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        # Convert results into readable output
        scores = result.to_pandas().iloc[0].to_dict()

        output = f"""
QUESTION:
{question}

GROUND TRUTH:
{reference}

GENERATED ANSWER:
{generated_answer}

RAGAS EVALUATION SCORES:
"""

        for metric, score in scores.items():
            try:
                output += f"\n{metric}: {float(score):.4f}"
            except (ValueError, TypeError):
                output += f"\n{metric}: {score}"

        return output

    except Exception as e:
        return f"Evaluation error: {str(e)}"


# -------------------- GRADIO UI --------------------

with gr.Blocks() as demo:

    qa_chain_state = gr.State(None)

    gr.Markdown("# Chat with Your PDF")

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

        label = gr.Textbox(label="Upload Status")

    chatbot = gr.Chatbot(
        label="Talk to the Document"
    )

    msg = gr.Textbox(
        label="Ask a Question",
        placeholder="Enter your question here..."
    )

    upload.click(
        fn=load_pdf_into_vectorstore,
        inputs=[file],
        outputs=[qa_chain_state, label]
    )

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, qa_chain_state],
        outputs=[chatbot, msg]
    )

    # -------------------- EVALUATION SECTION --------------------

    gr.Markdown("---")
    gr.Markdown("# RAGAS Evaluation")

    gr.Markdown(
        "Enter a test question and its correct answer "
        "to evaluate your RAG system."
    )

    evaluation_question = gr.Textbox(
        label="Test Question",
        placeholder="Example: What is the main topic of this document?"
    )

    ground_truth = gr.Textbox(
        label="Ground Truth (Correct Answer)",
        placeholder="Enter the expected correct answer..."
    )

    evaluate_button = gr.Button(
        "Evaluate RAG",
        variant="primary"
    )

    evaluation_output = gr.Textbox(
        label="RAGAS Evaluation Results",
        lines=15
    )

    evaluate_button.click(
        fn=evaluate_rag,
        inputs=[
            qa_chain_state,
            evaluation_question,
            ground_truth
        ],
        outputs=evaluation_output
    )


demo.launch(debug=True)
