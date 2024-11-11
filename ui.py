import argparse

import streamlit as st

import config
import llm
from data import Conversation, Message


def display_chat(chat_history: Conversation) -> None:
    for message in chat_history.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def update_chat(role: str, content: str) -> None:
    st.session_state.chat_history.messages.append(Message(role=role, content=content))


def ui_main(args: argparse.Namespace, llm_chat: llm.LLMChat, chat_history: Conversation) -> None:
    st.title(f"Chat with {args.model_name}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    # Display conversation
    display_chat(st.session_state.chat_history)

    # Get user input
    if prompt := st.chat_input("Your message:"):
        print(st.session_state.chat_history)
        # Add user message to chat history
        update_chat(role="user", content=prompt)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            if args.stream_generations:
                stream = llm_chat.generate_response(st.session_state.chat_history)
                response = st.write_stream(stream)
            else:
                response = llm_chat.generate_response(st.session_state.chat_history)
                st.write(response)

        # Add assistant response to chat history
        update_chat(role="assistant", content=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Assistant")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        choices=config.llm_model_name_choices,
        help="The name of the model to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    parser.add_argument(
        "--stream_generations", action="store_true", help="Flag to switch on streaming, only possible for OpenAI models"
    )
    args = parser.parse_args()

    # Initialize conversation and LLM
    chat_history: Conversation = Conversation(messages=[])
    match args.model_name:
        case "gpt-4o" | "gpt-4" | "gpt-3.5":
            llm_chat = llm.OpenAIChat(
                model_name=args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=args.seed,
                stream_generations=args.stream_generations,
            )
        case name if "llama" in name:
            llm_chat = llm.LlamaChat(
                model_path=args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=args.seed,
            )

    ui_main(args, llm_chat, chat_history)
