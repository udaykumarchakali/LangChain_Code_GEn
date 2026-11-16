
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

#Load API key
load_dotenv()

#Create tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract second number from first number."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        return "Cannot divide by zero"
    return a / b

#Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

#Create agent
agent = create_agent(
    model=llm,
    tools=[add, subtract, multiply, divide],
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the available tools to perform calculations. "
        "Explain the final answer in simple English."
    )
)

# Run agent workflow
while True:
    question = input("\nAsk a math question (or type exit): ")

    if question.lower() == "exit":
        print("Agent stopped.")
        break

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    # Display final response
    print("\nAgent:", response["messages"][-1].content)
