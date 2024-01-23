"""
Example on using openAI's retrieval tool with the Assistant API.

see: https://platform.openai.com/docs/assistants/overview

You'll need GPT-4 for this to work.
"""
import helpers
import llm

if __name__ == "__main__":
    helpers.set_cwd()
    helpers.load_secrets(secrets_file=".env")

    # set up an assistant that has the prompting guide available for retrieval
    _assistant = llm.create_assistant(
        name="test-assistant",
        instructions="You are a prompt engineering assistant, "
        "you answer questions based on the provided retrieval context.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_paths=["./code-samples/openai-prompting-guide.txt"],
    )

    # start a conversation, first message is the user asking about prompt engineering
    _thread = llm.client.beta.threads.create()
    llm.client.beta.threads.messages.create(
        thread_id=_thread.id,
        role="user",
        content="Tell me about prompt engineering.",
    )
    run = llm.client.beta.threads.runs.create(
        thread_id=_thread.id,
        assistant_id=_assistant.id,
    )

    # wait a bit here until the run is completed
    run = llm.client.beta.threads.runs.retrieve(thread_id=_thread.id, run_id=run.id)
    print("run status: ", run.status)
    assert run.status == "completed"

    messages = llm.client.beta.threads.messages.list(
      thread_id=_thread.id
    )
    print("user: ", messages.data[1].content[0].text.value)
    print("assistant: ", messages.data[0].content[0].text.value)

    # open question: how do you make the question-wait-response loop generic?
    # how does the thing perform with multiple documents, how would you test
    # its performance?
