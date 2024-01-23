def test_helpers():
    import os
    import helpers

    helpers.set_cwd()
    helpers.ensure_requirements()
    helpers.load_secrets(secrets_file=".env.template")

    assert os.getenv("OPENAI_API_TYPE") in ("azure", "openai")


def test_llm():
    import helpers
    import llm

    helpers.set_cwd()
    helpers.ensure_requirements()
    helpers.load_secrets(secrets_file=".env.template")

    llm.create_assistant(
        name="test-assistant",
        instructions="You are a test assistant, "
                     "you only reply with 'whooooo yeah buddy'",
        model="gpt-35-turbo-16k",
    )