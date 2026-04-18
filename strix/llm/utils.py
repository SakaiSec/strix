import html
import inspect
import re
from typing import Any

from strix.tools.registry import get_tool_by_name, get_tool_param_schema


_INVOKE_OPEN = re.compile(r'<invoke\s+name=["\']([^"\']+)["\']>')
_PARAM_NAME_ATTR = re.compile(r'<parameter\s+name=["\']([^"\']+)["\']>')
_PARAM_NAME_ATTR_MALFORMED = re.compile(r'<parameter\s+name=(["\'])([^"\'>]+)>')
_FUNCTION_CALLS_TAG = re.compile(r"</?function_calls>")
_STRIP_TAG_QUOTES = re.compile(r"<(function|parameter)\s*=\s*([^>]*?)>")


def normalize_tool_format(content: str) -> str:
    """Convert alternative tool-call XML formats to the expected one.

    Handles:
      <function_calls>...</function_calls>  → stripped
      <invoke name="X">                     → <function=X>
      <parameter name="X">                  → <parameter=X>
      </invoke>                             → </function>
      <function="X">                        → <function=X>
      <parameter="X">                       → <parameter=X>
    """
    content = _PARAM_NAME_ATTR.sub(r"<parameter=\1>", content)
    content = _PARAM_NAME_ATTR_MALFORMED.sub(r"<parameter=\2>", content)

    if "<invoke" in content or "<function_calls" in content:
        content = _FUNCTION_CALLS_TAG.sub("", content)
        content = _INVOKE_OPEN.sub(r"<function=\1>", content)
        content = content.replace("</invoke>", "</function>")

    return _STRIP_TAG_QUOTES.sub(
        lambda m: f"<{m.group(1)}={m.group(2).strip().strip(chr(34) + chr(39))}>", content
    )


STRIX_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "claude-opus-4.6": "anthropic/claude-opus-4-6",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5.4": "openai/gpt-5.4",
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "glm-5": "openrouter/z-ai/glm-5",
    "glm-4.7": "openrouter/z-ai/glm-4.7",
}


def resolve_strix_model(model_name: str | None) -> tuple[str | None, str | None]:
    """Resolve a strix/ model into names for API calls and capability lookups.

    Returns (api_model, canonical_model):
    - api_model: openai/<base> for API calls (Strix API is OpenAI-compatible)
    - canonical_model: actual provider model name for litellm capability lookups
    Non-strix models return the same name for both.
    """
    if not model_name or not model_name.startswith("strix/"):
        return model_name, model_name

    base_model = model_name[6:]
    api_model = f"openai/{base_model}"
    canonical_model = STRIX_MODEL_MAP.get(base_model, api_model)
    return api_model, canonical_model


def _truncate_to_first_function(content: str) -> str:
    if not content:
        return content

    function_starts = [
        match.start() for match in re.finditer(r"<function=|<invoke\s+name=", content)
    ]

    if len(function_starts) >= 2:
        second_function_start = function_starts[1]

        return content[:second_function_start].rstrip()

    return content


def parse_tool_invocations(content: str) -> list[dict[str, Any]] | None:
    content = normalize_tool_format(content)
    content = fix_incomplete_tool_call(content)

    tool_invocations: list[dict[str, Any]] = []

    fn_regex_pattern = r"<function=([^>]+)>\n?(.*?)</function>"
    fn_param_regex_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"

    fn_matches = re.finditer(fn_regex_pattern, content, re.DOTALL)

    for fn_match in fn_matches:
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)

        param_matches = re.finditer(fn_param_regex_pattern, fn_body, re.DOTALL)

        args = {}
        parse_error: str | None = None
        for param_match in param_matches:
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            param_value = html.unescape(param_value)
            args[param_name] = param_value

        if not args:
            parse_error = _build_raw_body_parse_error(fn_name, fn_body)
            raw_body_args = _extract_body_fallback_args(fn_name, fn_body)
            if raw_body_args:
                args = raw_body_args

        tool_invocation: dict[str, Any] = {"toolName": fn_name, "args": args}
        if not args and parse_error:
            tool_invocation["parseError"] = parse_error

        tool_invocations.append(tool_invocation)

    return tool_invocations if tool_invocations else None


def _extract_body_fallback_args(tool_name: str, fn_body: str) -> dict[str, str]:
    raw_body = html.unescape(fn_body).strip()
    if not raw_body:
        return {}

    required_param = _get_single_required_parameter(tool_name)
    if required_param:
        return {required_param: raw_body}

    param_schema = get_tool_param_schema(tool_name)
    if not param_schema:
        return {}

    required_params = sorted(param_schema.get("required", set()))
    if len(required_params) != 1:
        return {}

    return {required_params[0]: raw_body}


def _build_raw_body_parse_error(tool_name: str, fn_body: str) -> str | None:
    raw_body = html.unescape(fn_body).strip()
    if not raw_body:
        return None

    param_schema = get_tool_param_schema(tool_name)
    if not param_schema:
        return None

    required_params = sorted(param_schema.get("required", set()))
    if len(required_params) <= 1:
        return None

    required_list = ", ".join(required_params)
    return (
        f"Tool '{tool_name}' could not parse raw body text into explicit parameters. "
        f"Provide tagged arguments for required parameter(s): {required_list}"
    )


def _get_single_required_parameter(tool_name: str) -> str | None:
    tool_func = get_tool_by_name(tool_name)
    if tool_func is None:
        return None

    required_params = [
        param.name
        for param in inspect.signature(tool_func).parameters.values()
        if param.name != "agent_state"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Parameter.empty
    ]

    if len(required_params) != 1:
        return None

    return required_params[0]


def fix_incomplete_tool_call(content: str) -> str:
    """Fix incomplete tool calls by adding missing closing tag.

    Handles both ``<function=…>`` and ``<invoke name="…">`` formats.
    """
    has_open = "<function=" in content or "<invoke " in content
    count_open = content.count("<function=") + content.count("<invoke ")
    has_close = "</function>" in content or "</invoke>" in content
    if has_open and count_open == 1 and not has_close:
        content = content.rstrip()
        content = content + "function>" if content.endswith("</") else content + "\n</function>"
    return content


def format_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    xml_parts = [f"<function={tool_name}>"]

    for key, value in args.items():
        xml_parts.append(f"<parameter={key}>{value}</parameter>")

    xml_parts.append("</function>")

    return "\n".join(xml_parts)


def clean_content(content: str) -> str:
    if not content:
        return ""

    content = normalize_tool_format(content)
    content = fix_incomplete_tool_call(content)

    tool_pattern = r"<function=[^>]+>.*?</function>"
    cleaned = re.sub(tool_pattern, "", content, flags=re.DOTALL)

    incomplete_tool_pattern = r"<function=[^>]+>.*$"
    cleaned = re.sub(incomplete_tool_pattern, "", cleaned, flags=re.DOTALL)

    partial_tag_pattern = r"<f(?:u(?:n(?:c(?:t(?:i(?:o(?:n(?:=(?:[^>]*)?)?)?)?)?)?)?)?)?$"
    cleaned = re.sub(partial_tag_pattern, "", cleaned)

    hidden_xml_patterns = [
        r"<inter_agent_message>.*?</inter_agent_message>",
        r"<agent_completion_report>.*?</agent_completion_report>",
    ]
    for pattern in hidden_xml_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)

    return cleaned.strip()
