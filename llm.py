# azure_openai_35turbo.py

"""Test Microsoft Azure's ChatCompletion endpoint"""
import os
import openai
import helpers
import pydantic
from typing import Optional

# ensures requirements are installed & secrets have been loaded
helpers.init()

openai.api_type = os.environ["OPENAI_API_TYPE"]
if openai.api_type == "azure":
    openai.api_version = "2023-05-15"
    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2023-12-01-preview",
    )

elif openai.api_type == "openai":
    client = openai.Client(api_key=os.environ["OPENAI_KEY"])
else:
    raise ValueError(f"Unknown API type {openai.api_type}")


class Message(pydantic.BaseModel):
    """
    pydantic model for 'message'. 'Messages' going into ChatCompletion is a list-of-dicts,
    like [{"role": "user", "content": "Hello, world."}, ...]
    """

    # extra keys will make the API go boom
    model_config = pydantic.ConfigDict(extra="forbid")
    role: str
    content: str

    # validate the role to be one of the allowed types
    # the model probably won't know what to do with anything it wasn't trained on
    @pydantic.field_validator("role")
    def validate_role(cls, v):
        if v not in ("user", "system", "assistant"):
            raise ValueError(
                f"Invalid role {v}, must be one of 'user', 'system', 'assistant' "
                f"to avoid the LLM behaving in unexpected ways."
            )
        return v


class Function(pydantic.BaseModel):
    """
    pydantic model for 'function'.

    example usage

    "tools": [{
      "type": "function",
      "function": {
        "name": "getCurrentWeather",
        "description": "Get the weather in location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
            "unit": {"type": "string", "enum": ["c", "f"]}
          },
          "required": ["location"]
        }
      }
    },
    """

    name: str
    description: str
    # paramters is nested, muust have "type" and "properties" keys
    parameters: dict[str, dict[str, str]]

    # validate the parameters to have the required keys
    @pydantic.field_validator("parameters")
    def validate_parameters(cls, v):
        if "type" not in v:
            raise ValueError(
                f"Invalid parameters {v}, must have a 'type' key"
            )
        if "properties" not in v:
            raise ValueError(
                f"Invalid parameters {v}, must have a 'properties' key"
            )
        return v


class Tool(pydantic.BaseModel):
    """
    pydantic model for 'tool'. This is a dict, with in the simple cases just
    {"type": "code_interpreter"} or {"type": "retrieval"}; a more complex case
    is the custom function {"type": "function", ...} which needs an extra Function
    field
    """

    type: str  # required

    # optional, Function type
    function: Function = None

    # validate the type to be one of the allowed types
    @pydantic.field_validator("type")
    def validate_type(cls, v):
        if v not in ("code_interpreter", "retrieval", "function"):
            raise ValueError(
                f"Invalid type {v}, must be one of "
                f"'code_interpreter', 'retrieval', 'function' "
            )
        return v


def get_chat_completion(
    messages: list[Message],
    model: str,
    tools: Optional[list[Tool]] = None,
    max_tokens: int = 128,
    temperature: float = 0.5,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
):
    """
    Get a chat completion from the OpenAI Python SDK.

    Note that:
    - OpenAI from OpenAI uses `model="gpt-3.5-turbo"`!
    - OpenAI from Azure uses `model="‹deployment name›"`! ⚠️

    When using Azure, you need to set the model variable to the deployment name
      you chose when you deployed a model to your resource. Try something like
      "gpt-35-turbo-16k"; check the Azure portal for the name of your deployment.
    When using OpenAI, you need to set the model variable to something listed
      in their docs, such as "gpt-3.5-turbo".
    """
    return client.chat.completions.create(
        # dump messages into a list of dicts
        messages=[m.model_dump() for m in messages],
        model=model,
        tools=[t.model_dump() for t in tools],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


c = get_chat_completion(
    # messages=[{"role": "user", "content": "Hello, world."}],
    messages=[
        Message(
            role="user",
            content="What is the temperature in Delft, NL?",
            tools=Tool(
                type="function",
                function=Function(
                    name="weather",
                    description="Get the weather",
                    parameters={"location": "Delft, NL"},
                ),
            ),
        )
    ],
    model="gpt-35-turbo-16k",
)
