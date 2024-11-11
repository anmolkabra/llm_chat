import argparse

import streamlit as st

import config
from data import Conversation, Message
from llm import LlamaChat, OpenAIChat


def display_conv(conv: Conversation) -> None:
    for message in conv.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def ui_main(args: argparse.Namespace) -> None:
    st.title(f"Chat with {args.model_name}")

    # Initialize conversation and LLM
    conv: Conversation = Conversation(messages=[])
    match args.model_name:
        case "gpt-4o" | "gpt-4" | "gpt-3.5":
            llm = OpenAIChat(
                model_name=args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=args.seed,
                stream=args.stream_generations,
            )
        case "llama3.2":
            llm = LlamaChat(
                model_path=args.model_name,
                max_retries=3,
                wait_seconds=2,
                temperature=0,
                seed=args.seed,
            )

    # Display conversation
    display_conv(conv)

    # Get user input
    if prompt := st.chat_input("Your message:"):
        # Add user message to chat history
        conv.messages.append(Message(role="user", content=prompt))

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            if args.stream_generations:
                stream = llm.generate_response(conv)
                response = st.write_stream(stream)
            else:
                response = llm.generate_response(conv)
                st.write(response)

        # Add assistant response to chat history
        conv.messages.append(Message(role="assistant", content=response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit Chat")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        choices=config.llm_model_name_choices,
        help="The name of the OpenAI model to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    parser.add_argument(
        "--stream_generations", store_action=True, help="Flag to switch on streaming, only possible for OpenAI models"
    )
    args = parser.parse_args()

    ui_main(args)
