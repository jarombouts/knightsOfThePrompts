# azure_openai_35turbo.py

"""Test Microsoft Azure's ChatCompletion endpoint"""
import os
import openai
import helpers

# ensures requirements are installed & secrets have been loaded
helpers.init()

openai.api_type = os.environ["OPENAI_API_TYPE"]
if openai.api_type == "azure":
    openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    openai.api_key = os.environ["AZURE_OPENAI_KEY"]
elif openai.api_type == "openai":
    openai.api_key = os.environ["OPENAI_KEY"]
else:
    raise ValueError(f"Unknown API type {openai.api_type}")

# Hello, world.
# In addition to the `api_*` properties above, mind the difference in arguments
# as well between OpenAI and Azure:
# - OpenAI from OpenAI uses `model="gpt-3.5-turbo"`!
# - OpenAI from Azure uses `engine="‹deployment name›"`! ⚠️
#   > You need to set the engine variable to the deployment name you chose when
#   > you deployed the GPT-35-Turbo or GPT-4 models.
#  This is the name of the deployment I created in the Azure portal on the resource.


def get_chat_completion(
    messages: list,
    engine: str | None = None,
    model: str | None = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
):
    """
    Get a chat completion from the OpenAI Python SDK.

    Note that:
    - OpenAI from OpenAI uses `model="gpt-3.5-turbo"`!
    - OpenAI from Azure uses `engine="‹deployment name›"`! ⚠️

    When using Azure, you need to set the engine variable to the deployment name
      you chose when you deployed a model to your resource. Try something like
      "gpt-35-turbo-16k"; check the Azure portal for the name of your deployment.
    When using OpenAI, you need to set the model variable to something listed
      in their docs, such as "gpt-3.5-turbo".
    """
    # only ONE of engine or model should be set
    if engine is not None and model is not None:
        raise ValueError(
            "Only one of engine (when using Azure) "
            "or model (when using OpenAI) should be set"
        )
    if engine is None and model is None:
        raise ValueError(
            "One of engine (when using Azure) "
            "or model (when using OpenAI) should be set"
        )
    return openai.ChatCompletion.create(
        engine=engine,
        prompt=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

c = get_chat_completion(
    messages=[{"role": "user", "content": "Hello, world."}],
    engine="gpt-35-turbo-16k",
)