import streamlit as st 
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import ChatMessageHistory
import os


# load environment
load_dotenv()

import os

print("OpenAI key:", bool(os.getenv("OPENAI_API_KEY")))
print("LangSmith key:", bool(os.getenv("LANGSMITH_API_KEY")))
print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))

model= ChatOpenAI(model="gpt-4o-mini")

# setup the session state
if "store" not in st.session_state:
    st.session_state.store={}

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "conversation_counter" not in st.session_state:
    st.session_state.conversation_counter =0
    

# create new converstion
def create_new_conversation():
    st.session_state.conversation_counter +=1
    
    new_conv_id = str(uuid.uuid4())
    
    st.session_state.conversations[new_conv_id]={
        "number":st.session_state.conversation_counter,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages":[]
    }
    
    st.session_state.current_conversation_id = new_conv_id
    
    
# create first converstion
if "current_conversation_id" not in st.session_state:
    create_new_conversation()
    

# get session history
def get_session_history(user_id:str, conversation_id:str)-> ChatMessageHistory:
    if (user_id, conversation_id) not in st.session_state.store:
        st.session_state.store[(user_id, conversation_id)] = ChatMessageHistory()
    return st.session_state.store[(user_id,conversation_id)]


# Create Prompt Template
prompt_template=ChatPromptTemplate.from_messages([  
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of you ability in {language}"
    ),
    MessagesPlaceholder(variable_name="mychat_history"),
    
    ("human","{input}")
])

# create chain
chain = prompt_template | model

# Add Meassage History
model_with_message_history= RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="mychat_history",
    history_factory_config=[
        
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            is_shared=True
        ),
        
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            is_shared=True
        )
    ]
    
)

# this handle the user question and Ai response to save in chat history
def handle_chat_input(prompt, config):
    current_conv= st.session_state.conversations[
        st.session_state.current_conversation_id
    ]
    
    # save user message in ccmh
    current_conv["messages"].append({
        "role":"user",
        "content":prompt
    })                   
    
    # get AI response
    
    result= model_with_message_history.invoke(
        {
            "input":prompt,
            "language":"English"
        },
        config=config
    )
    
    # Save Assistant Message
    current_conv["messages"].append({
        "role":"assistant",
        "content":result.content
    })
    

with st.sidebar:
    st.subheader("Conversations")

    if st.button("New Conversation"):
        create_new_conversation()
        st.rerun()
    # this stores conver ids conversation1,conversation1
    conversation_list = [
        (
            conv_id,
            f"Conversation {conv_data['number']}"
        )
        for conv_id, conv_data in st.session_state.conversations.items()
    ]

    conversation_ids = [conv[0] for conv in conversation_list]

    selected_conv = st.selectbox(
        "Select Conversation:",
        options=conversation_ids,
        format_func=lambda x: next(
            conv[1] for conv in conversation_list if conv[0] == x
        ),
        index=conversation_ids.index(
            st.session_state.current_conversation_id
        ),
        key="conversation_selector"
    )
# its return the new selected id as current cid
    if selected_conv != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_conv
        st.rerun()

    st.sidebar.divider()

    st.write(f"User ID: {st.session_state.user_id}")
    st.write(f"Current Conversation ID: {st.session_state.current_conversation_id}")


# get current converstion
current_conv=st.session_state.conversations[
    st.session_state.current_conversation_id
]
# chat input
if user_prompt:=st.chat_input("what's on your mind?"):
    
    config={
        "configurable":{
            "user_id":st.session_state.user_id,
            "conversation_id":st.session_state.current_conversation_id
        }
    }
    
    try:
        handle_chat_input(user_prompt,config)
    except Exception as e:
        st.error(f"Error:{e}")

# Display Converstion MessagesPlaceholder
for message in current_conv["messages"]:
    with st.chat_message(message["role"]):
        
        st.write(message["content"])  
    

    
 