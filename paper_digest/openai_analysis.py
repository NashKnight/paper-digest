"""OpenAI-backed structured paper analysis."""

from __future__ import annotations

import json
import os
import subprocess
from urllib.request import Request, urlopen

from .arxiv_client import Paper, PaperAnalysis
from .config import AnalysisConfig, DigestTemplate


class OpenAIAnalysisError(RuntimeError):
    """Raised when OpenAI analysis fails."""


RELEVANCE_LABELS = {
    "omni_duplex_core",
    "omni_duplex_related",
    "omni_simplex_related",
    "not_relevant",
}


def classify_paper_relevance_with_openai(
    config: AnalysisConfig,
    paper: Paper,
) -> tuple[str, str]:
    """Classify whether a keyword-matched paper belongs in the omni digest."""

    payload = _build_json_request_payload(
        config,
        instructions=(
            "You are filtering papers for a daily research digest about omni-modal "
            "real-time speech interaction. Classify the paper using only the title, "
            "metadata, and abstract. Labels: omni_duplex_core for full-duplex or "
            "simultaneous listen-and-speak speech agents; omni_duplex_related for "
            "closely related real-time streaming speech, voice-agent, barge-in, "
            "turn-taking, interruption, endpointing, or audio-language-model work; "
            "omni_simplex_related for one-way or turn-based omni/multimodal work, "
            "including audio stream processing, streaming audio understanding, "
            "video understanding, audio-visual reasoning, audio-visual temporal "
            "grounding, half-duplex dialogue, simplex speech dialogue, and "
            "cascaded voice assistants; not_relevant for unrelated uses of "
            "omni/duplex or wireless, radio, optical, networking, or generic "
            "multimodal work without audio, speech, video, or omni-model relevance. "
            "Return compact JSON only."
        ),
        input_text=_build_input(paper),
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": sorted(RELEVANCE_LABELS)},
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
        },
        schema_name="paper_relevance",
    )
    raw = _request_json(config, payload)
    label = _required_string(raw.get("label"), "relevance.label")
    reason = _first_string(
        raw,
        ("reason", "rationale", "explanation", "justification"),
        "relevance.reason",
    )
    if label not in RELEVANCE_LABELS:
        raise OpenAIAnalysisError(f"unsupported relevance label: {label}")
    return label, reason


def analyze_paper_with_openai(
    config: AnalysisConfig,
    paper: Paper,
    *,
    template: DigestTemplate = "default",
) -> PaperAnalysis:
    """Analyze a single paper with the OpenAI Responses API."""

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise OpenAIAnalysisError(
            f"analysis API key environment variable {config.api_key_env!r} is not set"
        )

    payload = _build_json_request_payload(
        config,
        instructions=_build_instructions(config, template=template),
        input_text=_build_input(paper),
        schema=_analysis_schema(),
        schema_name="paper_analysis",
    )
    raw_analysis = _request_json(config, payload, api_key=api_key)
    return _parse_paper_analysis(raw_analysis)


def _build_json_request_payload(
    config: AnalysisConfig,
    *,
    instructions: str,
    input_text: str,
    schema: dict[str, object],
    schema_name: str,
) -> dict[str, object]:
    if _uses_chat_completions(config.base_url):
        schema_prompt = (
            "Return only valid JSON matching this schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        return {
            "model": config.model,
            "messages": [
                {"role": "system", "content": f"{instructions}\n\n{schema_prompt}"},
                {"role": "user", "content": input_text},
            ],
            "stream": False,
            "max_tokens": config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

    payload: dict[str, object] = {
        "model": config.model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": config.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    if config.reasoning_effort != "none":
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def _request_json(
    config: AnalysisConfig,
    payload: dict[str, object],
    *,
    api_key: str | None = None,
) -> dict[str, object]:
    if api_key is None:
        api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise OpenAIAnalysisError(
            f"analysis API key environment variable {config.api_key_env!r} is not set"
        )

    request = Request(
        config.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            raw_payload = response.read()
    except OSError as exc:
        raw_payload = _request_json_with_curl(
            config,
            payload,
            api_key=api_key,
            original_error=exc,
        )

    response_json = _load_response_json(raw_payload)
    response_text = _extract_response_text(response_json)

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OpenAIAnalysisError(
            "OpenAI analysis response was not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise OpenAIAnalysisError("OpenAI analysis payload is invalid")
    return raw


def _request_json_with_curl(
    config: AnalysisConfig,
    payload: dict[str, object],
    *,
    api_key: str,
    original_error: OSError,
) -> bytes:
    command = [
        "curl",
        "-sS",
        "--connect-timeout",
        str(min(config.timeout_seconds, 30)),
        "--max-time",
        str(config.timeout_seconds),
        "-X",
        "POST",
        config.base_url,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {api_key}",
        "-d",
        json.dumps(payload),
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower()
        not in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
        }
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=env,
            timeout=config.timeout_seconds + 5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenAIAnalysisError(
            "failed to call OpenAI-compatible API with urllib "
            f"({original_error}) and curl fallback ({exc})"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OpenAIAnalysisError(
            "failed to call OpenAI-compatible API with urllib "
            f"({original_error}) and curl fallback exited "
            f"{completed.returncode}: {stderr}"
        )
    return completed.stdout


def _build_instructions(
    config: AnalysisConfig,
    *,
    template: DigestTemplate,
) -> str:
    template_hint = ""
    if template == "zh_daily_brief":
        template_hint = (
            " Prefer newsroom-style phrasing that reads naturally in a Chinese daily"
            " research briefing."
        )
    return (
        "You are writing concise research-digest notes. "
        "Use only the provided title, metadata, and abstract. "
        "Do not invent empirical claims or missing details. "
        "If the abstract does not support a point, say so cautiously. "
        f"Write every field in {config.language}. "
        "Keep each field compact and useful for a daily paper digest."
        f"{template_hint}"
    )


def _build_input(paper: Paper) -> str:
    authors = ", ".join(paper.authors) if paper.authors else "Unknown authors"
    categories = (
        ", ".join(paper.categories) if paper.categories else "Unknown categories"
    )
    return (
        f"Title: {paper.title}\n"
        f"Source: {paper.source}\n"
        f"Authors: {authors}\n"
        f"Categories: {categories}\n"
        f"Published: {paper.published_at.isoformat()}\n"
        f"Abstract URL: {paper.abstract_url}\n"
        f"Abstract:\n{paper.summary}"
    )


def _analysis_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conclusion": {"type": "string"},
            "contributions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "audience": {"type": "string"},
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "conclusion",
            "contributions",
            "audience",
            "limitations",
        ],
    }


def _load_response_json(payload: bytes) -> dict[str, object]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenAIAnalysisError("received malformed JSON from OpenAI") from exc

    if not isinstance(raw, dict):
        raise OpenAIAnalysisError("OpenAI response payload is invalid")

    error = raw.get("error")
    if isinstance(error, dict):
        message = error.get("message", "unknown error")
        raise OpenAIAnalysisError(f"OpenAI returned an error: {message}")

    status = raw.get("status")
    if isinstance(status, str) and status not in {"completed", "in_progress"}:
        raise OpenAIAnalysisError(
            f"OpenAI response did not complete successfully: {status}"
        )
    return raw


def _extract_response_text(raw: dict[str, object]) -> str:
    choices = raw.get("choices")
    if isinstance(choices, list):
        fragments: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                fragments.append(content.strip())
        if fragments:
            return "\n".join(fragments)

    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = raw.get("output")
    if not isinstance(output, list):
        raise OpenAIAnalysisError("OpenAI response did not include output content")

    fragments: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal":
            raise OpenAIAnalysisError("OpenAI refused to analyze the paper")

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            item_type = content_item.get("type")
            if item_type in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
            if item_type == "refusal":
                raise OpenAIAnalysisError("OpenAI refused to analyze the paper")

    if not fragments:
        raise OpenAIAnalysisError("OpenAI response did not include analysis text")
    return "\n".join(fragments)


def _uses_chat_completions(base_url: str) -> bool:
    return base_url.rstrip("/").endswith("/chat/completions")


def _parse_paper_analysis(raw: object) -> PaperAnalysis:
    if not isinstance(raw, dict):
        raise OpenAIAnalysisError("OpenAI analysis payload is invalid")

    conclusion = _required_string(raw.get("conclusion"), "analysis.conclusion")
    audience = _required_string(raw.get("audience"), "analysis.audience")
    contributions = _string_list(raw.get("contributions"), "analysis.contributions")
    limitations = _string_list(raw.get("limitations"), "analysis.limitations")

    return PaperAnalysis(
        conclusion=conclusion,
        contributions=contributions,
        audience=audience,
        limitations=limitations,
    )


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise OpenAIAnalysisError(f"{field_name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise OpenAIAnalysisError(f"{field_name} must not be empty")
    return normalized


def _first_string(
    raw: dict[str, object],
    field_names: tuple[str, ...],
    display_name: str,
) -> str:
    for field_name in field_names:
        value = raw.get(field_name)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    raise OpenAIAnalysisError(f"{display_name} must be a string")


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise OpenAIAnalysisError(f"{field_name} must be an array of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise OpenAIAnalysisError(f"{field_name} must contain only strings")
        normalized = " ".join(item.split())
        if normalized:
            result.append(normalized)
    return result
