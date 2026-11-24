
import streamlit as st
from main import ask_agent


st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 AI Agent")
st.caption("Powered by Ollama, LangChain, Tavily and Open-Meteo")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_question = st.chat_input(
    "Ask me anything..."
)


if user_question:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            try:
                answer = ask_agent(user_question)

                st.markdown(answer)

            except Exception as error:

                answer = f"Error: {error}"
                st.error(answer)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
