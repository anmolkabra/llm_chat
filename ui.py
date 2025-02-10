import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import streamlit as st
from PIL import Image

from _types import ChatSession, ContentImageMessage, ContentTextMessage, Conversation, Message
from llm import SUPPORTED_LLM_SERVERS, get_llm
from llm.common import LLMChat


def init_conv(add_init_image: bool = False) -> Conversation:
    content = []
    if add_init_image:
        image = Image.open("assets/Image.jpg")
        content.append(ContentImageMessage(image=image))
    messages: list[Message] = [] if not content else [Message(role="user", content=content, created_at=datetime.now())]
    return Conversation(messages=messages)


@st.cache_resource
def get_llm_chat(_args: argparse.Namespace) -> LLMChat:
    """
    Loads the LLM chat object based on the model name.
    Function is cached in streamlit so that the model is not reloaded on every streamlit ui repaint.

    Args:
        _args (argparse.Namespace): The parsed arguments.

    Returns:
        llm.LLMChat: The LLM chat object.
    """
    # Use _args instead of args so that streamlit does not hash the variable namespace object
    llm_kwargs = {
        "model_path": _args.model_path,
        "max_tokens": _args.max_tokens,
        "temperature": _args.temperature,
        "seed": _args.seed,
    }
    return get_llm(_args.server, _args.model_name, llm_kwargs)


def format_md_text(text: str) -> str:
    """
    Formats text for markdown display:
    - Encloses <think>...</think> in a blockquote.
    """
    # Replace <think>...</think> with blockquote
    text = text.replace("<think>", "<blockquote>").replace("</think>", "</blockquote>")
    return text


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
                        formatted_text = format_md_text(text)
                        st.markdown(formatted_text, unsafe_allow_html=True)
                    case ContentImageMessage(image=image):
                        st.image(image, use_container_width=True)


def update_chat(role: Literal["user", "assistant"], text: str) -> None:
    """
    Update the chat history in streamlit's session state with a new text message assigned to a role.

    Args:
        role (Literal["user", "assistant"]): Either "user" or "assistant".
        text (str): Content of the message.
    """
    new_message = Message(role=role, content=[ContentTextMessage(text=text)], created_at=datetime.now())
    st.session_state.chat_history.messages.append(new_message)


def clear_chat() -> None:
    """
    Clear the chat history in streamlit's session state.
    """
    st.session_state.chat_history = init_conv()


def load_chat_from_path(file_path: str) -> None:
    """
    Load chat history from a JSON file and update the chat history in streamlit's session state.

    Args:
        file_path (str): Path to the JSON file.
    """
    if not Path(file_path).expanduser().exists():
        st.error(f"File does not exist: {file_path}")
        return
    chat_session = ChatSession.load_from_path(file_path)
    st.session_state.chat_history = chat_session.conv


def save_chat_to_path(file_path: str) -> None:
    """
    Save chat history to a JSON file. If the file's directory does not exist, it will be created.

    Args:
        file_path (str): Path to the JSON file.
    """
    # If the file path directory does not exist, create it
    Path(file_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    # If the file path does not have a .json extension, add it
    if Path(file_path).suffix != ".json":
        file_path += ".json"
    chat_session = ChatSession(
        llm_name=st.session_state.llm_chat.model_name,
        llm_kwargs=st.session_state.llm_chat.model_kwargs,
        conv=st.session_state.chat_history,
    )
    chat_session.save_to_path(file_path)


def display_sidebar() -> None:
    """
    Display the sidebar:
    - Button for Clear chat
    - Widget for Load chat from file
    - Widget for Save chat to file
    """
    with st.sidebar:
        st.button("Clear chat", on_click=clear_chat)

        st.divider()

        st.subheader("Load chat")
        load_file_path = st.text_input("JSON file path for loading chat")
        st.button("Load chat", on_click=load_chat_from_path, args=(load_file_path.strip(),))

        st.divider()

        st.subheader("Save chat")
        save_file_path = st.text_input("JSON file path for saving chat")
        st.button("Save chat", on_click=save_chat_to_path, args=(save_file_path.strip(),))

        st.divider()


def ui_main(args: argparse.Namespace) -> None:
    """
    Main UI function for the chat application.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    st.title(f"Chat with {args.model_name}")

    llm_chat: LLMChat = st.session_state.llm_chat

    # Display sidebar
    display_sidebar()

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
            response = llm_chat.generate_response(st.session_state.chat_history)
            formatted_text = format_md_text(response)
            st.markdown(formatted_text, unsafe_allow_html=True)

        # Add assistant response to chat history
        update_chat(role="assistant", text=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Assistant")
    parser.add_argument(
        "--server",
        type=str,
        default="together",
        choices=SUPPORTED_LLM_SERVERS,
        help="The server to use for the assistant",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-Vision-Free",
        help="The name of the model to use",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model to use")
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum number of tokens for the model's response"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for the model's response generation"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    args = parser.parse_args()

    # Initialize conversation and LLM
    # TODO Add image to chat history -- some HF models complain
    # HACK Feed an image at the beginning. Otherwise llama complains
    # is_hf_model = args.model_name in llm.HuggingfaceChat.SUPPORTED_LLM_NAMES
    is_hf_model = False
    chat_history = init_conv(add_init_image=is_hf_model)
    llm_chat: LLMChat = get_llm_chat(args)

    if "llm_chat" not in st.session_state:
        st.session_state.llm_chat = llm_chat

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    ui_main(args)
