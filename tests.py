def test_helpers():
    import os
    import helpers

    helpers.set_cwd()
    helpers.ensure_requirements()
    helpers.load_secrets(secrets_file=".env.template")

    assert os.getenv("OPENAI_API_TYPE") in ("azure", "openai")
