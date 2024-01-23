"""Example usage of the 'llm' module to create a personalized chatbot."""

from llm import get_chat_completion, Message
from loguru import logger

if __name__ == "__main__":
    who = input("Who do you want to chat with?")
    messages = [
        Message(
            role="system",
            content=f"You are a chatbot impersonating {who}. "
            f"You remain in character at all times, not breaking immersion.",
        ),
        Message(
            role="assistant",
            content="I will initiate the conversation by greeting the user in a "
            "suitable, character-specific way.",
        ),
    ]
    while True:
        # main loop: get answer from model, ask user for reply, repeat.
        completion = get_chat_completion(
            messages, model="gpt-35-turbo-16k", max_tokens=1024
        )
        logger.info("\n" + completion.choices[0].message.content)
        messages.append(
            Message(
                role=completion.choices[0].message.role,
                content=completion.choices[0].message.content,
            )
        )
        messages.append(Message(role="user", content=input("You: ")))

        # open question: how can you make sure the model doesn't run out of context
        # when the chat history grows, with the `messages` object having too much info?
