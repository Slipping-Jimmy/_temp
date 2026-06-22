#!/usr/bin/env python3
"""NHI smart chat reference retrieval helpers for current data builders."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib.parse import urljoin

import requests


DEFAULT_BASE_URL = "https://nhismartchat.nhi.gov.tw"


def url_from_env(name: str, default_path: str, **values: str) -> str:
    template = os.getenv(name, "").strip()
    if not template:
        base_url = os.getenv("NHI_SMARTCHAT_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/"
        return urljoin(base_url, default_path.lstrip("/")).format(**values)
    return template.format(**values)


def fetch_chat(question: str) -> tuple[str, str, str, Any]:
    """Call the NHI chatbot API and return the generated answer plus contexts."""
    res = requests.post(
        url_from_env("NHI_SMARTCHAT_CONVERSATION_URL", "/api/-/conversations")
    )
    res.raise_for_status()
    conv_id = res.text.strip().strip('"')

    chat_id = str(uuid.uuid4())
    payload = {"question": question, "chat_id": chat_id, "status": "common"}
    headers = {"Content-Type": "application/json"}

    stream_res = requests.post(
        url_from_env(
            "NHI_SMARTCHAT_STREAM_URL_TEMPLATE",
            "/api/chatbot/conversations/{conversation_id}/stream",
            conversation_id=conv_id,
            chat_id=chat_id,
        ),
        json=payload,
        headers=headers,
        stream=True,
    )
    stream_res.raise_for_status()

    full_response = ""
    for line in stream_res.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data:"):
            continue
        data_str = decoded[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "text" in data:
            full_response += data["text"]
        elif "msg" in data:
            full_response += data["msg"]

    ctx_res = requests.get(
        url_from_env(
            "NHI_SMARTCHAT_CONTEXTS_URL_TEMPLATE",
            "/api/chatbot/conversations/{conversation_id}/chats/{chat_id}/rephrased-contexts",
            conversation_id=conv_id,
            chat_id=chat_id,
        )
    )
    ctx_res.raise_for_status()
    try:
        contexts: Any = ctx_res.json()
    except json.JSONDecodeError:
        contexts = ctx_res.text

    return conv_id, chat_id, full_response, contexts


def format_information(contexts: Any) -> str:
    """Format retrieved contexts into the reference text used by SFT builders."""
    if isinstance(contexts, list):
        info = ""
        for context in contexts:
            if isinstance(context, dict) and "title" in context and "content" in context:
                info += (
                    f"問題:{context['title']}\n"
                    f"回答:{context['content']}\n"
                    "===============SECTION===============\n"
                )
            elif isinstance(context, dict) and "question" in context and "answer" in context:
                info += (
                    f"問題:{context['question']}\n"
                    f"回答:{context['answer']}\n"
                    "===============SECTION===============\n"
                )
            elif isinstance(context, dict) and "content" in context:
                info += f"{context['content']}\n===============SECTION===============\n"
            else:
                info += (
                    json.dumps(context, ensure_ascii=False)
                    + "\n===============SECTION===============\n"
                )
        return info

    if (
        isinstance(contexts, dict)
        and "data" in contexts
        and isinstance(contexts["data"], list)
    ):
        return format_information(contexts["data"])

    return str(contexts)
