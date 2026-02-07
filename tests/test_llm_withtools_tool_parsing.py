# This file is adapted from https://github.com/jennyzzt/dgm.

from types import SimpleNamespace

from llm_withtools import check_for_tool_use, process_tool_call


def test_check_for_tool_use_openrouter_malformed_json_fallback_to_raw():
    bad_args = '{"command": "echo hi"'
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="bash", arguments=bad_args),
                        )
                    ]
                )
            )
        ]
    )
    logs = []
    tool_use = check_for_tool_use(
        response, model="deepseek-chat", logging=lambda x: logs.append(str(x))
    )

    assert tool_use["tool_name"] == "bash"
    assert tool_use["tool_input"] == bad_args
    assert any("Failed to parse tool arguments for bash" in msg for msg in logs)


def test_check_for_tool_use_openrouter_literal_dict_fallback():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id="call_2",
                            function=SimpleNamespace(
                                name="bash", arguments="{'command': 'echo hi'}"
                            ),
                        )
                    ]
                )
            )
        ]
    )

    tool_use = check_for_tool_use(response, model="deepseek-chat")
    assert tool_use["tool_input"] == {"command": "echo hi"}


def test_check_for_tool_use_openai_ignores_non_function_calls():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="message"),
            SimpleNamespace(type="reasoning"),
        ]
    )

    tool_use = check_for_tool_use(response, model="gpt-4o")
    assert tool_use is None


def test_process_tool_call_accepts_scalar_input_for_command_tool():
    def bash(command):
        return f"ran:{command}"

    tools_dict = {"bash": {"function": bash}}
    result = process_tool_call(tools_dict, "bash", "echo ok")
    assert result == "ran:echo ok"


def test_process_tool_call_dict_path_still_works():
    def adder(a, b=0):
        return a + b

    tools_dict = {"adder": {"function": adder}}
    result = process_tool_call(tools_dict, "adder", {"a": 3, "b": 4})
    assert result == 7
