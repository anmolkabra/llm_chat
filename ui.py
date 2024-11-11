import argparse

import requests
import streamlit as st
from PIL import Image

import config
import llm
from data import ContentImageMessage, ContentTextMessage, Conversation, Message


def display_chat(chat_history: Conversation) -> None:
    for message in chat_history.messages:
        with st.chat_message(message.role):
            for content in message.content:
                match content:
                    case ContentTextMessage(text=text):
                        st.markdown(text)
                    case ContentImageMessage(image=image):
                        st.image(image, use_container_width=True)


def update_chat(role: str, text: str) -> None:
    st.session_state.chat_history.messages.append(Message(role=role, content=[ContentTextMessage(text=text)]))


def ui_main(args: argparse.Namespace, llm_chat: llm.LLMChat, chat_history: Conversation) -> None:
    st.title(f"Chat with {args.model_name}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    # Display conversation
    display_chat(st.session_state.chat_history)

    # Get user input
    if prompt := st.chat_input("Your message:"):
        # Add user message to chat history
        update_chat(role="user", text=prompt)

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
        update_chat(role="assistant", text=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Assistant")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        choices=config.llm_model_name_choices,
        help="The name of the model to use",
    )
    parser.add_argument("--model_local_path", type=str, default=None, help="Path to the model to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    parser.add_argument(
        "--stream_generations", action="store_true", help="Flag to switch on streaming, only possible for OpenAI models"
    )
    args = parser.parse_args()

    # Initialize conversation and LLM
    # HACK Feed an image at the beginning. Otherwise llama complains
    url = "https://upload.wikimedia.org/wikipedia/commons/7/78/Image.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    init_message = Message(role="user", content=[ContentTextMessage(text="Hello!"), ContentImageMessage(image=image)])
    chat_history: Conversation = Conversation(messages=[init_message])
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
                model_path=args.model_local_path or args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0.01,
                seed=args.seed,
            )

    # FIXME llama loads on every paint??
    ui_main(args, llm_chat, chat_history)
