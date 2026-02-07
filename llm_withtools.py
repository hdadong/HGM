# This file is adapted from https://github.com/jennyzzt/dgm.

import ast
import copy
import inspect
import json
import re
from time import time

import anthropic
import backoff
import openai

from llm import create_client
from tools import load_all_tools


CLAUDE_MODEL = 'deepseek-chat'
OPENAI_MODEL = 'deepseek-chat'


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            tool_function = tools_dict[tool_name]["function"]

            # Standard path: object-like tool input.
            if isinstance(tool_input, dict):
                return tool_function(**tool_input)

            # Fallback: scalar input for single-arg tools or command-like tools.
            signature = inspect.signature(tool_function)
            required_params = [
                p
                for p in signature.parameters.values()
                if p.default is inspect._empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]

            if len(required_params) == 1:
                return tool_function(**{required_params[0].name: tool_input})

            if "command" in signature.parameters:
                return tool_function(command=tool_input)

            return (
                f"Error executing tool '{tool_name}': tool_input must be an object "
                f"for this tool, got {type(tool_input).__name__}"
            )
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
    ),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice, logging=None, max_retry=3
):
    try:
        if model.startswith("o") or "gpt" in model.lower():
            response = client.responses.create(
                model=model,
                # reasoning={"effort": "low"},
                input=[
                    {
                        "role": "system",
                        "content": "You are the best coder in the world!",
                    }
                ]
                + messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False,
            )
        else:
            response = client.chat.completions.create(
                model=client.models.list().data[0].id
                if "vllm" in model.lower()
                else model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are the best coder in the world!",
                    }
                ]
                + messages,
                tool_choice=tool_choice,
                tools=tools,
                parallel_tool_calls=False,
            )
        return response
    except Exception as e:
        logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(
                client, model, messages, tools, tool_choice, logging, max_retry - 1
            )

        # Hitting the context window limit
        if "Input is too long for requested model" in str(e):
            pass

        raise  # Re-raise the exception after logging


def _log_parse_failure(logging, message):
    if logging:
        try:
            logging(message)
        except Exception:
            pass


def _truncate_text(text, max_len=300):
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _parse_tool_input(raw, logging=None, context="tool_input"):
    # Already structured.
    if isinstance(raw, (dict, list, int, float, bool)) or raw is None:
        return raw

    text = str(raw).strip()
    if not text:
        _log_parse_failure(logging, f"{context} is empty")
        return None

    # Preferred parser for provider tool-call arguments.
    try:
        return json.loads(text)
    except Exception as e_json:
        json_err = str(e_json)

    # Best-effort fallback for quasi-Python literals.
    try:
        return ast.literal_eval(text)
    except Exception as e_lit:
        _log_parse_failure(
            logging,
            (
                f"Failed to parse {context} as JSON/literal; using raw string. "
                f"json_error={json_err}; literal_error={str(e_lit)}; "
                f"snippet={_truncate_text(text)}"
            ),
        )
        return text


def check_for_tool_use(response, model="", logging=None):
    """
    Checks if the response contains a tool call.
    """

    if model.startswith("o") or "gpt" in model.lower():
        # OpenAI, check for tool_calls in response
        tool_call = None
        for call in response.output:
            if call.type == "function_call":
                tool_call = call
                break

        if tool_call:
            return {
                "tool_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "tool_input": _parse_tool_input(
                    tool_call.arguments,
                    logging=logging,
                    context=f"tool arguments for {tool_call.name}",
                ),
            }

    else:
        if (
            response.choices[0].message.tool_calls is None
            or len(response.choices[0].message.tool_calls) == 0
        ):
            return False
        call = response.choices[0].message.tool_calls[0]
        return {
            "tool_id": call.id,
            "tool_name": call.function.name,
            "tool_input": _parse_tool_input(
                call.function.arguments,
                logging=logging,
                context=f"tool arguments for {call.function.name}",
            ),
        }

    # No tool use found
    return None


def convert_tool_info(tool_info, model=None):
    """
    Converts tool_info from Claude format to the given model's format.
    """
    if "vllm" in model.lower():
        required = [
            val_name for val_name in tool_info["input_schema"]["properties"].keys()
        ]
        return {
            "type": "function",
            "function": {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_info["input_schema"]["properties"],
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    elif model.startswith("o") or "gpt" in model.lower():

        def add_additional_properties(d):
            if isinstance(d, dict):
                if "properties" in d:
                    d["additionalProperties"] = False
                for k, v in d.items():
                    add_additional_properties(v)

        add_additional_properties(tool_info["input_schema"])
        for p in tool_info["input_schema"]["properties"].keys():
            if not p in tool_info["input_schema"]["required"]:
                tool_info["input_schema"]["required"].append(p)
                t = copy.deepcopy(tool_info["input_schema"]["properties"][p]["type"])
                if isinstance(t, str):
                    tool_info["input_schema"]["properties"][p]["type"] = [t, "null"]
                elif isinstance(t, list):
                    tool_info["input_schema"]["properties"][p]["type"] = t + ["null"]

        return {
            "type": "function",
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["input_schema"],
            "strict": True,
        }
    else:
        required = [
            val_name for val_name in tool_info["input_schema"]["properties"].keys()
        ]
        return {
            "type": "function",
            "function": {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_info["input_schema"]["properties"],
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


def chat_with_agent_openai(
    msg,
    model=OPENAI_MODEL,
    msg_history=None,
    logging=print,
    max_llm_calls=1000,  # Maximum number of LLM calls to make
    timeout=3600,
):
    start_time = time()
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": msg,
                }
            ],
        }
    ]
    separator = "=" * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    try:
        # Create client
        client, client_model = create_client(model)

        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
        tools = [
            convert_tool_info(tool["info"], model=client_model) for tool in all_tools
        ]

        for i in range(max_llm_calls):
            if timeout * 0.9 < time() - start_time:
                logging("Timeout reached, stopping further LLM calls.")
                return new_msg_history, i
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )
            logging(f"Tool Response: {response}")
            tool_use = check_for_tool_use(
                response, model=client_model, logging=logging
            )
            new_msg_history += response.output
            if not tool_use:
                return new_msg_history, i + 1
            # Process tool call
            tool_name = tool_use["tool_name"]
            tool_input = tool_use["tool_input"]
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            new_msg_history.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_use["tool_id"],
                    "output": tool_result,
                }
            )

    except Exception:
        pass

    return new_msg_history, max_llm_calls


def chat_with_agent_open_router(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    max_llm_calls=1000,  # Maximum number of LLM calls to make
    timeout=3600,
):
    start_time = time()
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = [{"role": "user", "content": msg}]
    separator = "=" * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")
    try:
        # Create client
        client, client_model = create_client(model)
        # Load all tools
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
        tools = [
            convert_tool_info(tool["info"], model=client_model) for tool in all_tools
        ]
        for i in range(max_llm_calls):
            if timeout * 0.9 < time() - start_time:
                logging("Timeout reached, stopping further LLM calls.")
                return new_msg_history, i
            # Process tool call
            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )

            new_msg_history.append(response.choices[0].message)
            logging(f"Tool Response: {response}")
            # Check for next tool use
            tool_use = check_for_tool_use(
                response, model=client_model, logging=logging
            )
            if not tool_use:
                return new_msg_history, i + 1
            tool_name = tool_use["tool_name"]
            tool_input = tool_use["tool_input"]
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)
            tool_use["content"] = tool_result

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            # Get tool response
            new_msg_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use["tool_id"],
                    "name": tool_use["tool_name"],
                    "content": f"{tool_result}",
                }
            )

    except Exception as e:
        logging(f"Error in chat_with_agent_open_router: {str(e)}")

    return new_msg_history, max_llm_calls


def convert_msg_history_openai(msg_history):
    """
    Convert OpenAI-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        role = ""
        content = ""
        if isinstance(msg, dict):
            if "role" in msg.keys():
                role = msg["role"]
            else:
                role = "user"
            if "content" in msg.keys():
                content = msg["content"]
            else:
                content = "Tool Result: " + msg.get("output", "")

        else:
            role = "assistant"
            content = str(msg)

        new_msg_history.append({"role": role, "content": content})

    return new_msg_history


def convert_msg_history_open_router(msg_history):
    """
    Convert OpenRouter-style message history into a generic format.
    """
    new_msg_history = []

    for msg in msg_history:
        if not isinstance(msg, dict):
            msg = dict(msg)
        role = msg.get("role", "")
        if "content" in msg.keys():
            if role == "tool":
                content = "Tool Result: " + msg["content"]
            else:
                content = msg["content"]
        else:
            content = f"Function: {msg['tool_calls'][0].name}\nArguments: {msg['tool_calls'][0].function.arguments}"

        new_msg_history.append({"role": role, "content": content})

    return new_msg_history


def convert_msg_history(msg_history, model=None):
    """
    Convert message history from the model-specific format to a generic format.
    """
    if model.startswith("o") or "gpt" in model.lower():
        return convert_msg_history_openai(msg_history)
    else:
        return convert_msg_history_open_router(msg_history)


def chat_with_agent(
    msg,
    model=CLAUDE_MODEL,
    msg_history=None,
    logging=print,
    convert=False,  # Convert the message history to a generic format, so that msg_history can be used across models
    max_llm_calls=1000,  # Maximum number of LLM calls to make
    timeout=3600,
):
    if msg_history is None:
        msg_history = []

    if model.startswith("o") or "gpt" in model.lower():
        # OpenAI models
        new_msg_history, n_llm_calls = chat_with_agent_openai(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
        )
        new_msg_history = msg_history + new_msg_history

    else:
        new_msg_history, n_llm_calls = chat_with_agent_open_router(
            msg,
            model=model,
            msg_history=msg_history,
            logging=logging,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
        )
        new_msg_history = msg_history + new_msg_history

    return new_msg_history, n_llm_calls


if __name__ == "__main__":
    # Test the tool calling functionality
    msg = "First create the current directory. Then implement a function that returns the current directory and save it in the directory just created. Finally call the function and return the result. In the end, summarize what you did."
    model = "vllm-qwenS-10.109.17.7"
    history, _ = chat_with_agent(msg, model=model, max_llm_calls=2)
    from utils.eval_utils import msg_history_to_report

    print(msg_history_to_report("hgm", history, model=model))
    # history = convert_msg_history(history, model)
    # chat_with_agent(msg, model, history, max_llm_calls=2)
