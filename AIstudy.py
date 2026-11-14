
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


# Create AI Study Assistant chain
def create_AI_study_Assistant():

    llm = ChatOpenAI(model="gpt-4o-mini")

    AI_Study_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced AI Study Assistant
                specializing in AI Engineering and Development.
                Explain concepts clearly in Markdown format."""
            ),
            (
                "human",
                """Explain the student's topic or doubt.

                - Explain in ultra-easy English.
                - Give one real-life example.
                - Provide code with a simple explanation when needed.

                Student's question: {question}
                Selected topic: {topic}
                Explanation level: {level}
                Output type: {output_type}
                """
            )
        ]
    )

    return AI_Study_prompt | llm | StrOutputParser()


# Configure Streamlit page
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide"
)

# Main title
st.title("Student AI Study Assistant")


# Sidebar
with st.sidebar:

    st.header("Study Settings")

    topic = st.selectbox(
        "Select a topic",
        ["Python", "AI", "ML", "DL", "LangChain"]
    )

    level = st.selectbox(
        "Explanation level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    output_type = st.selectbox(
        "Output type",
        ["Explanation", "Code Example", "Quiz"]
    )


# Question input
question = st.text_input("Enter your question")

analyze_button = st.button("Chat with AI")


# Generate answer
if analyze_button:

    if not question.strip():
        st.warning("Please enter your question.")

    else:  
        with st.spinner("Preparing your answer..."):

            try:
                chain = create_AI_study_Assistant()

                inputs = {
                    "question": question,
                    "topic": topic,
                    "level": level,
                    "output_type": output_type
                }

                chat_result = chain.invoke(inputs)

                st.markdown(chat_result)

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info(
        "select you topics and level, type to Chat with AI to get started."
    )
