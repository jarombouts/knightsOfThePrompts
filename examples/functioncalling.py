"""
Example showing how you can have the LLM request function calls based on user input.

We take an example of a banking assistant with one skill (modyfing user accounts)
and two intents (change address and change phone number).

The LLM automatically detects if the user needs the "modify user account" skill,
and changes the active skillset from "general" to "modify user account".

When "modify user account" is active, the LLM will automatically detect if the user
wants to change the address or the phone number, and will emit parameters that can
go into appropriate functions.
"""
import llm

import pydantic
import json
from typing import Optional


# First define some functions that the model is allowed to call
class ChangeAddress(pydantic.BaseModel):
    """
    Changes the address for a user account.

    Before calling this function, you must have looked up the user account ID by
        calling the LookupUser function, or the user must be asked to provide the user
        account ID themselves. Also you must have asked the user to provide the new
        address.
    """

    user_account_id: str = pydantic.Field(description="The user account ID")
    new_address: str = pydantic.Field(description="The new address for this user")


class ChangePhoneNumber(pydantic.BaseModel):
    """
    Changes the phone number for a user account.

    Before calling this function, you must have looked up the user account ID by
      calling the LookupUser function, or the user must be asked to provide the user
      account ID themselves. Also you must have asked the user to provide the new
      phone number.
    """

    user_account_id: str = pydantic.Field(description="The user account ID")
    new_phone_number: str = pydantic.Field(
        description="The new phone number for this user"
    )


class LookupUser(pydantic.BaseModel):
    """
    Resolves a user by name, bank account number, or address.

    Before calling this function, you must know or ask the user to provide at least one
      of the following:
    - The name of the user
    - The bank account number of the user
    - The address of the user
    """

    name: str = pydantic.Field(
        description="The name of the user to look up", default=None
    )
    bank_account_number: str = pydantic.Field(
        description="The bank account number of the user to look up", default=None
    )
    address: str = pydantic.Field(
        description="The address of the user to look up", default=None
    )

    # validate that one or more of the fields is set
    @pydantic.model_validator(mode="after")
    def check_at_least_one_field(self):
        if not any([self.name, self.bank_account_number, self.address]):
            raise ValueError(
                "Must specify at least one of name, bank_account_number, or address"
            )


class RequestMoreInformation(pydantic.BaseModel):
    """
    Requests more information from the user.

    When the LLM wants to call this function, it will emit a JSON containing
    the information that it wants to know.
    """

    information: str = pydantic.Field(
        description="The information that the LLM wants to gather from the user"
    )


def dump_function_schema(s):
    return json.dumps(s.model_json_schema(), indent=2)


def create_openai_tools(function_schemas: list):
    """
    Given a list of function schemas
    create an openai-formatted list of function specifications

    """
    function_spec = [
        {"type": "function", "function": f.model_json_schema()}
        for f in function_schemas
    ]
    # convert title field into name field, nest properties field in parameters field.
    # add function.parameters.type = "object" to each function.
    # move function.required into function.parameters.required
    for f in function_spec:
        f["function"]["name"] = f["function"]["title"]
        f["function"]["parameters"] = {"properties": f["function"]["properties"]}
        f["function"]["parameters"]["type"] = "object"
        if "required" in f["function"]:
            f["function"]["parameters"]["required"] = f["function"]["required"]
        else:
            f["function"]["parameters"]["required"] = []

    return function_spec


def process_tool_calls(tool_calls: list):
    for t in tool_calls:
        ... # process tool calls here

    return tool_calls


if __name__ == "__main__":
    # This main loop simulates a convo
    messages = [
        llm.Message(
            role="system",
            content="You are a banking assistant. You are tasked with greeting the user and figuring out what they want to do.",
        ),
        llm.Message(
            role="assistant",
            content="I will initiate the conversation by greeting the user in a kind, "
            "light-hearted yet professional demeanor. I will not call any functions at "
            "this point in the conversation; I will only call a function when I am sure"
            " that the user wants to undertake an action, and after the user has "
            "provided me with all the information I need to call the function.",
        ),
    ]

    # bot greets user without access to tools
    initial_greeting = llm.get_chat_completion(
        messages=messages,
        model="gpt-35-turbo-16k",
    )
    print("assistant: " + initial_greeting.choices[0].message.content)
    messages.append(
        llm.Message(
            role="assistant", content=initial_greeting.choices[0].message.content
        )
    )

    # user says they want to change their address
    messages.append(
        llm.Message(
            role="user",
            content="I want to change my address.",
        )
    )
    print("user: " + messages[-1].content)

    # the llm will decide which function to call
    llm_tool_choice = llm.get_chat_completion(
        messages=messages,
        model="gpt-35-turbo-16k",
        tools=create_openai_tools(
            [ChangeAddress, ChangePhoneNumber, LookupUser, RequestMoreInformation]
        ),
    )
    tool_call_results = process_tool_calls( # you need to expand handling tool calls here
        tool_calls=llm_tool_choice.choices[0].message.tool_calls
    )
    # ... and handle the case where there are no tool calls but instead just messages
    # see https://platform.openai.com/docs/guides/function-calling for more info
    for r in tool_call_results:
        print(f"inserting result of tool call to {r.function}")
        messages.append(
            llm.Message(
                role="assistant",
                content=json.dumps(
                    {
                        "tool_call_id": "...",
                        "role": "tool",
                        "name": "...",
                        "content": "...",
                    }
                ),
            )
        )
    llm_response_after_tool_call = llm.get_chat_completion(
        messages=messages,
        model="gpt-35-turbo-16k",
    )
    print("assistant: " + llm_response_after_tool_call.choices[0].message.content)