
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


# Create the financial analysis chain
def create_financial_chain():

    llm = ChatOpenAI(model="gpt-4o-mini")
    # .from_messages() this is a method used to create a chat prompt with multiple message types.
    financial_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced financial advisor in India
                who specializes in personal finance management.
                Analyze the customer's financial situation and provide
                detailed insights in a clear Markdown format."""
            ),
            (
                "human",
                """Perform an initial financial analysis for the client in India:

                - Monthly Income: {monthly_income}
                - Monthly Expenses: {monthly_expenses}
                - Current Savings: {current_savings}

                Create a comprehensive assessment of their financial health.

                Include these sections:
                - Current Financial Health Assessment
                - Cash Flow Analysis
                - Savings Potential
                - Risk Capacity Evaluation"""
            )
        ]
    )

    return financial_analysis_prompt | llm | StrOutputParser()


# Configure the Streamlit page
# using .set_page_config method created the set up
st.set_page_config(
    page_title="Financial Health Analysis",
    page_icon="💰",
    layout="wide"
)

# Display the main title
st.title("Personal Financial Health Analysis")


# Create the sidebar
with st.sidebar:

    st.header("Enter Your Financial Details")

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0,
        value=150000
    )

    monthly_expenses = st.number_input(
        "Monthly Expenses (₹)",
        min_value=0,
        value=80000
    )

    current_savings = st.number_input(
        "Current Savings (₹)",
        min_value=0,
        value=500000
    )

    analyze_button = st.button("Analyze Financial Health")


# Analyze the financial details
if analyze_button:

    with st.spinner("Analyzing your financial health..."):

        try:
            chain = create_financial_chain()

            inputs = {
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "current_savings": current_savings
            }

            analysis_result = chain.invoke(inputs)

            st.markdown(analysis_result)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info(
        "Enter your financial details in the sidebar "
        "and click 'Analyze Financial Health' to get started."
    )
