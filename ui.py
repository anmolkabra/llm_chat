import argparse
from pathlib import Path
from typing import Literal

import streamlit as st
from PIL import Image

import config
import llm
from data import ContentImageMessage, ContentTextMessage, Conversation, Message


def init_conv(add_init_image: bool = False) -> Conversation:
    content = [ContentTextMessage(text="Hello!")]
    if add_init_image:
        image = Image.open("assets/Image.jpg")
        content.append(ContentImageMessage(image=image))
    return Conversation(messages=[Message(role="user", content=content)])


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
        _args.model_name in config.supported_llm_model_names
    ), f"Model name must be one of {config.supported_llm_model_names}"

    match _args.model_name:
        case model_name if model_name in config.GPT_MODEL_NAMES:
            llm_chat = llm.OpenAIChat(
                model_name=model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=_args.seed,
                stream_generations=_args.stream_generations,
            )
        case model_name if model_name in config.LLAMA_MODEL_NAMES:
            llm_chat = llm.LlamaChat(
                model_path=_args.model_local_path or model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0.01,
                seed=_args.seed,
            )
        case model_name if model_name in config.TOGETHER_MODEL_NAMES:
            llm_chat = llm.TogetherChat(
                model_name=model_name.lstrip("together:"), # Remove the prefix "together:"
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=_args.seed,
                stream_generations=_args.stream_generations,
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
    chat_history = Conversation.load_from_path(file_path)
    st.session_state.chat_history = chat_history


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
    conv: Conversation = st.session_state.chat_history
    conv.save_to_path(file_path)


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

    llm_chat: llm.LLMChat = st.session_state.llm_chat

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
        default="together:meta-llama/Llama-Vision-Free",
        choices=config.supported_llm_model_names,
        help="The name of the model to use",
    )
    parser.add_argument("--model_local_path", type=str, default=None, help="Path to the model to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    parser.add_argument(
        "--stream_generations", action="store_true", help="Flag to switch on streaming, only possible for OpenAI models"
    )
    args = parser.parse_args()

    # Initialize conversation and LLM
    is_model_llama = args.model_name in config.LLAMA_MODEL_NAMES
    chat_history = init_conv(add_init_image=is_model_llama) # HACK Feed an image at the beginning. Otherwise llama complains
    llm_chat: llm.LLMChat = get_llm_chat(args)

    if "llm_chat" not in st.session_state:
        st.session_state.llm_chat = llm_chat

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = chat_history

    ui_main(args)
