import argparse
import os

import streamlit as st
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


class OpenAIChat:
    def __init__(self, model_name: str, stop_after_attempts: int, wait_seconds: int, temperature: float, seed: int):
        self.model_name = model_name
        self.stop_after_attempts = stop_after_attempts
        self.wait_seconds = wait_seconds
        self.temperature = temperature
        self.seed = seed
        self.client = OpenAI()
    
    def generate_response(self, prompt: str) -> str:
        # Wrap retry params inside generate_response
        # @retry(stop=stop_after_attempt(self.stop_after_attempts), wait=wait_fixed(self.wait_seconds))
        def _call_api(prompt: str) -> str:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                seed=self.seed,
            )
            return completion.choices[0].message
        
        message = _call_api(prompt)
        return message


def ui_main(args: argparse.Namespace) -> None:
    st.title("Chat with OpenAI GPT-4")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    openai_chat = OpenAIChat(
        model_name=args.model_name,
        stop_after_attempts=3, 
        wait_seconds=2, 
        temperature=0,
        seed=args.seed,
    )

    user_input = st.text_input("You: ", "")

    if st.button("Send"):
        if user_input:
            st.session_state.messages.append(f"You: {user_input}")
            response = openai_chat.generate_response(user_input)
            st.session_state.messages.append(f"{args.model_name}: {response}")

    for message in st.session_state.messages:
        st.write(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit OpenAI Chat")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="The name of the OpenAI model to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the model")
    args = parser.parse_args()

    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    ui_main(args)


