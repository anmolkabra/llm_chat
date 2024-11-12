import argparse
from typing import Literal

import streamlit as st
from PIL import Image

import config
import llm
from data import ContentImageMessage, ContentTextMessage, Conversation, Message


@st.cache_resource
def get_llm_chat(_args: argparse.Namespace) -> llm.LLMChat:
    """
    Loads the LLM chat object based on the model name.
    Function is cached in streamlit so that the model is not reloaded on every rerun.

    Args:
        _args (argparse.Namespace): The parsed arguments.

    Returns:
        llm.LLMChat: The LLM chat object.
    """
    # Use _args instead of args so that streamlit does not hash the variable namespace object
    assert (
        _args.model_name in config.llm_model_name_choices
    ), f"Model name must be one of {config.llm_model_name_choices}"

    match _args.model_name:
        case "gpt-4o" | "gpt-4" | "gpt-3.5":
            llm_chat = llm.OpenAIChat(
                model_name=_args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=_args.seed,
                stream_generations=_args.stream_generations,
            )
        case name if "llama" in name.lower():
            llm_chat = llm.LlamaChat(
                model_path=_args.model_local_path or _args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0.01,
                seed=_args.seed,
            )

    return llm_chat


def display_chat(chat_history: Conversation) -> None:
    """
    Display the chat history in the chat message container.

    Args:
        chat_history (Conversation): Conversation history object.
    """
    for message in chat_history.messages:
        with st.chat_message(message.role):
            for content in message.content:
                match content:
                    case ContentTextMessage(text=text):
                        st.markdown(text)
                    case ContentImageMessage(image=image):
                        st.image(image, use_container_width=True)


def update_chat(role: Literal["user", "assistant"], text: str) -> None:
    """
    Update the chat history in streamlit's session state with a new text message assigned to a role.

    Args:
        role (Literal["user", "assistant"]): Either "user" or "assistant".
        text (str): Content of the message.
    """
    st.session_state.chat_history.messages.append(Message(role=role, content=[ContentTextMessage(text=text)]))


def ui_main(args: argparse.Namespace) -> None:
    """
    Main UI function for the chat application.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    st.title(f"Chat with {args.model_name}")

    llm_chat: llm.LLMChat = st.session_state.llm_chat

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
    image = Image.open("assets/Image.jpg")
    init_message = Message(role="user", content=[ContentTextMessage(text="Hello!"), ContentImageMessage(image=image)])
    chat_history: Conversation = Conversation(messages=[init_message])
    llm_chat: llm.LLMChat = get_llm_chat(args)

    if "llm_chat" not in st.session_state:
        st.session_state.llm_chat = llm_chat

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    ui_main(args)
