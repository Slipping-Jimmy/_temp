#!/usr/bin/env python3
"""Replay historical negative-feedback questions against the current chatbot.

The script reads the monthly feedback Excel files in this directory, calls the
current chatbot SSE API, fetches retrieval contexts, and writes resumable CSV and
JSONL outputs for later LLM-as-Judge analysis.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import re
import sys
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import requests
from urllib3.exceptions import InsecureRequestWarning

try:
    import openpyxl
except ImportError:  # pragma: no cover - optional dependency
    openpyxl = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling.nhi_reference_api import format_information  # noqa: E402


DEFAULT_INPUT_GLOB = "testing/智能客服DMZ_不滿意回饋處理明細_202*.xlsx"
DEFAULT_OUTPUT_CSV = Path("testing/output/replayed_feedback_202511_202604.csv")
DEFAULT_OUTPUT_JSONL = Path("testing/output/replayed_feedback_202511_202604.jsonl")

DEFAULT_BASE_URL = "https://nhismartchattest.intra.nhi.gov.tw"
DEFAULT_CONVERSATION_URL = f"{DEFAULT_BASE_URL}/api/-/conversations"
DEFAULT_SSE_URL_TEMPLATE = (
    f"{DEFAULT_BASE_URL}/api/-/conversations/{{conversation_id}}/sse"
)
DEFAULT_CONTEXTS_URL_TEMPLATE = (
    f"{DEFAULT_BASE_URL}/api/-/conversations/"
    "{conversation_id}/chats/{chat_id}/rephrased-contexts"
)

QUESTION_COL = "民眾問題"
OLD_ANSWER_COL = "模型回答"
BAD_REASON_COL = "倒讚原因"
BAD_COMMENT_COL = "倒讚評論"
CATEGORY_COL = "狀況分類"
FINAL_CONFIRMED_COL = "組室最終確認正確回覆"
FINETUNED_ANSWER_COL = "微調後模型回答"


CSV_FIELDNAMES = [
    "source_month",
    "source_id",
    "case_number",
    "asked_at",
    "question",
    "old_model_answer",
    "bad_feedback_reason",
    "bad_feedback_comment",
    "feedback_category",
    "final_confirmed_answer",
    "finetuned_answer",
    "new_answer",
    "new_retrieval_text",
    "status",
    "error",
]

XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def normalize_header(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def find_column(headers: list[object], needle: str) -> int | None:
    normalized_needle = normalize_header(needle)
    for index, header in enumerate(headers):
        if normalized_needle in normalize_header(header):
            return index
    return None


def cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value).strip()


def excel_serial_to_datetime(value: float) -> dt.datetime:
    # Excel's Windows epoch has the historical 1900 leap-year bug; this offset
    # matches openpyxl's default behavior for modern serial values.
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=value)


def column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("main:si", XML_NS):
        parts = [node.text or "" for node in item.findall(".//main:t", XML_NS)]
        strings.append("".join(parts))
    return strings


def workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_root.findall("pkgrel:Relationship", XML_NS)
    }

    sheets: list[tuple[str, str]] = []
    for sheet in workbook_root.findall("main:sheets/main:sheet", XML_NS):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{{{XML_NS['rel']}}}id")
        target = rel_targets.get(rel_id or "")
        if not target:
            continue
        sheet_path = "xl/" + target.lstrip("/")
        sheets.append((name, sheet_path))
    return sheets


def read_xlsx_rows_stdlib(path: Path) -> list[tuple[str, list[tuple[object, ...]]]]:
    sheets: list[tuple[str, list[tuple[object, ...]]]] = []
    with zipfile.ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        for sheet_name, sheet_path in workbook_sheets(zf):
            root = ET.fromstring(zf.read(sheet_path))
            rows: list[tuple[object, ...]] = []
            for row in root.findall(".//main:sheetData/main:row", XML_NS):
                values: list[object] = []
                for cell in row.findall("main:c", XML_NS):
                    ref = cell.attrib.get("r", "")
                    idx = column_index(ref)
                    while len(values) <= idx:
                        values.append(None)

                    cell_type = cell.attrib.get("t", "")
                    value_node = cell.find("main:v", XML_NS)
                    inline_node = cell.find("main:is", XML_NS)
                    raw_value = value_node.text if value_node is not None else None

                    if cell_type == "s" and raw_value is not None:
                        value: object = shared_strings[int(raw_value)]
                    elif cell_type == "inlineStr" and inline_node is not None:
                        value = "".join(
                            node.text or "" for node in inline_node.findall(".//main:t", XML_NS)
                        )
                    elif raw_value is None:
                        value = ""
                    else:
                        try:
                            numeric = float(raw_value)
                            value = int(numeric) if numeric.is_integer() else numeric
                        except ValueError:
                            value = raw_value

                    values[idx] = value

                rows.append(tuple(values))
            sheets.append((sheet_name, rows))
    return sheets


def iter_xlsx_sheets(path: Path) -> Iterable[tuple[str, Iterable[tuple[object, ...]]]]:
    if openpyxl is not None:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            yield sheet.title, sheet.iter_rows(values_only=True)
        return

    for sheet_name, rows in read_xlsx_rows_stdlib(path):
        yield sheet_name, iter(rows)


def month_from_path(path: Path) -> str:
    match = re.search(r"(20\d{4})\d{2}_20\d{6}", path.name)
    if match:
        return match.group(1)
    match = re.search(r"(20\d{4})", path.name)
    return match.group(1) if match else ""


def month_in_range(month: str, start_month: str, end_month: str) -> bool:
    return bool(month) and start_month <= month <= end_month


def row_value(row: tuple[object, ...], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return cell_text(row[index])


def date_row_value(row: tuple[object, ...], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    value = row[index]
    if isinstance(value, (int, float)):
        return excel_serial_to_datetime(float(value)).isoformat(sep=" ")
    return cell_text(value)


def iter_feedback_rows(
    input_glob: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for file_path in sorted(Path().glob(input_glob) if not glob.has_magic(input_glob) else map(Path, glob.glob(input_glob))):
        month = month_from_path(file_path)
        if not month_in_range(month, start_month, end_month):
            continue

        for sheet_name, rows_iter in iter_xlsx_sheets(file_path):
            rows = iter(rows_iter)
            headers = list(next(rows, []) or [])
            if not headers:
                continue

            question_idx = find_column(headers, QUESTION_COL)
            if question_idx is None:
                continue

            index_map = {
                "case_number": find_column(headers, "編號"),
                "asked_at": find_column(headers, "提問日期"),
                "old_model_answer": find_column(headers, OLD_ANSWER_COL),
                "bad_feedback_reason": find_column(headers, BAD_REASON_COL),
                "bad_feedback_comment": find_column(headers, BAD_COMMENT_COL),
                "feedback_category": find_column(headers, CATEGORY_COL),
                "final_confirmed_answer": find_column(headers, FINAL_CONFIRMED_COL),
                "finetuned_answer": find_column(headers, FINETUNED_ANSWER_COL),
            }

            for excel_row_number, row in enumerate(rows, start=2):
                question = row_value(row, question_idx)
                if not question:
                    continue
                source_id = f"{month}:{file_path.name}:{sheet_name}:{excel_row_number}"
                records.append(
                    {
                        "source_month": month,
                        "source_id": source_id,
                        "case_number": row_value(row, index_map["case_number"]),
                        "asked_at": date_row_value(row, index_map["asked_at"]),
                        "question": question,
                        "old_model_answer": row_value(row, index_map["old_model_answer"]),
                        "bad_feedback_reason": row_value(row, index_map["bad_feedback_reason"]),
                        "bad_feedback_comment": row_value(row, index_map["bad_feedback_comment"]),
                        "feedback_category": row_value(row, index_map["feedback_category"]),
                        "final_confirmed_answer": row_value(row, index_map["final_confirmed_answer"]),
                        "finetuned_answer": row_value(row, index_map["finetuned_answer"]),
                    }
                )
    return records


def load_completed(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {
            row.get("source_id", "")
            for row in reader
            if row.get("source_id")
            and row.get("status") == "ok"
            and row.get("new_answer", "").strip()
        }


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def remove_existing_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def parse_conversation_id(response: requests.Response) -> str:
    text = response.text.strip()
    try:
        body = response.json()
    except ValueError:
        return text.strip('"')

    if isinstance(body, str):
        return body.strip('"')
    if isinstance(body, dict):
        for key in ("conversation_id", "conversationId", "conv_id", "id"):
            value = body.get(key)
            if value:
                return str(value).strip('"')
        data = body.get("data")
        if isinstance(data, dict):
            for key in ("conversation_id", "conversationId", "conv_id", "id"):
                value = data.get(key)
                if value:
                    return str(value).strip('"')
        if isinstance(data, str):
            return data.strip('"')

    raise RuntimeError(f"Cannot parse conversation id from response: {text[:500]}")


def create_conversation(
    session: requests.Session,
    url: str,
    user_id: str,
    from_source: str,
    ip: str,
    timeout: float,
    verify_tls: bool,
) -> str:
    payload = {"user_id": user_id, "from_source": from_source, "ip": ip}
    response = session.post(url, json=payload, timeout=timeout, verify=verify_tls)
    response.raise_for_status()
    return parse_conversation_id(response)


def parse_sse_answer(response: requests.Response) -> str:
    answer_parts: list[str] = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if data_text in {"[DONE]", "[END]"}:
            break
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            answer_parts.append(data_text)
            continue
        if not isinstance(data, dict):
            if isinstance(data, (str, int, float)):
                answer_parts.append(str(data))
            continue
        for key in ("text", "msg", "content", "answer"):
            value = data.get(key)
            if value:
                answer_parts.append(str(value))
                break
    return "".join(answer_parts).strip()


def call_chatbot(
    question: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    session = requests.Session()
    conversation_id = create_conversation(
        session=session,
        url=args.conversation_url,
        user_id=args.user_id,
        from_source=args.from_source,
        ip=args.ip,
        timeout=args.timeout,
        verify_tls=args.verify_tls,
    )
    chat_id = str(uuid.uuid4())
    sse_url = args.sse_url_template.format(conversation_id=conversation_id)
    payload = {"chat_id": chat_id, "question": question, "status": args.status}
    response = session.post(
        sse_url,
        json=payload,
        timeout=args.timeout,
        stream=True,
        verify=args.verify_tls,
    )
    response.raise_for_status()
    answer = parse_sse_answer(response)

    contexts: Any = None
    retrieval_text = ""
    if not args.skip_retrieval:
        contexts_url = args.contexts_url_template.format(
            conversation_id=conversation_id,
            chat_id=chat_id,
        )
        ctx_response = session.get(
            contexts_url,
            timeout=args.timeout,
            verify=args.verify_tls,
        )
        ctx_response.raise_for_status()
        try:
            contexts = ctx_response.json()
        except ValueError:
            contexts = ctx_response.text
        retrieval_text = format_information(contexts).strip()

    return answer, retrieval_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay 202511-202604 feedback questions against the current chatbot."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--start-month", default="202511")
    parser.add_argument("--end-month", default="202604")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override all chatbot API URL templates, e.g. https://host or http://host:port.",
    )
    parser.add_argument("--conversation-url", default=DEFAULT_CONVERSATION_URL)
    parser.add_argument("--sse-url-template", default=DEFAULT_SSE_URL_TEMPLATE)
    parser.add_argument("--contexts-url-template", default=DEFAULT_CONTEXTS_URL_TEMPLATE)
    parser.add_argument("--user-id", default="a123456")
    parser.add_argument("--from-source", default="web_service")
    parser.add_argument("--ip", default="192.168.1.121")
    parser.add_argument("--status", default="common")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the initial conversation API connectivity check.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop after this many consecutive API errors. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.base_url:
        base_url = args.base_url.rstrip("/")
        args.conversation_url = f"{base_url}/api/-/conversations"
        args.sse_url_template = f"{base_url}/api/-/conversations/{{conversation_id}}/sse"
        args.contexts_url_template = (
            f"{base_url}/api/-/conversations/"
            "{conversation_id}/chats/{chat_id}/rephrased-contexts"
        )
    return args


def main() -> None:
    args = parse_args()
    if not args.verify_tls:
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

    records = iter_feedback_rows(args.input_glob, args.start_month, args.end_month)
    if args.limit:
        records = records[: args.limit]

    completed = load_completed(args.output_csv) if args.resume else set()
    print(f"Loaded cases: {len(records)}")
    print(f"Already completed: {len(completed)}")
    if args.dry_run:
        for record in records[:10]:
            print(record["source_id"], record["question"][:80])
        print("dry-run only; no API calls were made")
        return
    if not args.resume:
        remove_existing_outputs([args.output_csv, args.output_jsonl])

    if not args.skip_preflight:
        print(f"Preflight conversation API: {args.conversation_url}", flush=True)
        try:
            preflight_session = requests.Session()
            preflight_conversation_id = create_conversation(
                session=preflight_session,
                url=args.conversation_url,
                user_id=args.user_id,
                from_source=args.from_source,
                ip=args.ip,
                timeout=args.timeout,
                verify_tls=args.verify_tls,
            )
            print(f"Preflight OK, conversation_id={preflight_conversation_id}", flush=True)
        except Exception as exc:
            raise SystemExit(
                "Preflight failed. Conversation API is not reachable or returned an "
                f"unexpected response: {exc}\n"
                "Fix --conversation-url / network routing first, or pass "
                "--skip-preflight if you intentionally want to continue."
            ) from exc

    status_counts = {"ok": 0, "error": 0, "skipped": 0}
    error_samples: list[str] = []
    consecutive_errors = 0

    for index, base_record in enumerate(records, start=1):
        source_id = base_record["source_id"]
        if source_id in completed:
            print(f"[{index}/{len(records)}] skip completed: {source_id}", flush=True)
            status_counts["skipped"] += 1
            continue

        result = dict(base_record)
        result.update(
            {
                "new_answer": "",
                "new_retrieval_text": "",
                "status": "ok",
                "error": "",
            }
        )

        print(f"[{index}/{len(records)}] replay: {base_record['question'][:80]}", flush=True)
        try:
            answer, retrieval_text = call_chatbot(base_record["question"], args)
            result["new_answer"] = answer
            result["new_retrieval_text"] = retrieval_text
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            error_message = f"[{index}/{len(records)}] ERROR {source_id}: {exc}"
            print(error_message, flush=True)
            if len(error_samples) < 5:
                error_samples.append(error_message)
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        output_record = {field: result.get(field, "") for field in CSV_FIELDNAMES}

        append_csv(args.output_csv, output_record)
        append_jsonl(args.output_jsonl, output_record)
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
            print(
                f"Stopping after {consecutive_errors} consecutive errors. "
                "Use --max-consecutive-errors 0 to disable this guard.",
                flush=True,
            )
            break

    print(f"CSV saved/appended: {args.output_csv}")
    print(f"JSONL saved/appended: {args.output_jsonl}")
    print(
        "Summary:",
        f"ok={status_counts.get('ok', 0)}",
        f"error={status_counts.get('error', 0)}",
        f"skipped={status_counts.get('skipped', 0)}",
    )
    if error_samples:
        print("Error samples:")
        for sample in error_samples:
            print(f"  {sample}")


if __name__ == "__main__":
    main()
