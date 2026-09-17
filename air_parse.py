#!/usr/bin/env python3
"""Parse an Apple Intelligence Report (AIR) JSON into readable CSV, HTML, TXT, Markdown and JSON.

Standalone reimplementation. Stdlib only: no network calls, no subprocesses, no
dynamic imports. The report is opened read-only and every output goes to an
explicit directory. Run `--demo` to exercise the whole pipeline on synthetic data.

Exit codes: 0 success, 1 input error, 2 output exists without --force,
3 refused output location.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_mod
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

MAX_INPUT_BYTES = 512 * 1024 * 1024

# Writing into these would modify the operating system rather than the user's own
# data, so the tool refuses regardless of what --out says.
BLOCKED_OUTPUT_PREFIXES = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/Library/Apple",
    "/private/var/db",
    "/private/var/folders",
)

# Report values vary by OS version; match on substrings instead of exact equality.
ORIGIN_CLASSES = {
    "OnDevice": "on-device",
    "PrivateCloudCompute": "cloud",
    "Unknown": "unknown",
}

USE_CASE_TRIGGERS = {
    "summarization": "Summarize",
    "synopsis": "Summarize",
    "text_summarizer": "Summarize",
    "textComposition.OpenEndedTone": "Compose",
    "textComposition.OpenEndedToneQueryResponseV2": "Compose",
    "writingTools.compose": "Compose",
    "textComposition.TakeawaysTransform": "Key Points",
    "takeaways_transform": "Key Points",
    "textComposition.BulletsTransform": "List",
    "bullets_transform": "List",
    "textComposition.TablesTransform": "Table",
    "tables_transform": "Table",
    "GenerativeAssistant.knowledge": "Knowledge",
    "GenerativeAssistant.knowledgeFallback": "Knowledge",
    "GenerativeAssistant.composition": "Compose",
    "GenerativeAssistant.visualIntelligenceCamera": "Visual Intelligence",
    "VisualGeneration.GenerativePlayground": "Image Generation",
    "memoryCreation": "Memory Creation",
    "photos_memories": "Memory Creation",
}

CLIENT_APPS = {
    "com.apple.siri": "Siri",
    "com.apple.WritingToolsUIService": "Writing Tools",
    "com.apple.GenerativePlaygroundApp": "Image Playground",
    "com.apple.mobileslideshow": "Photos",
    "com.apple.mobilesafari": "Safari",
    "com.apple.mail": "Mail",
    "com.apple.mobilemail": "Mail",
    "com.apple.MobileSMS": "Messages",
    "com.apple.Notes": "Notes",
}

CSV_COLUMNS = [
    "Timestamp (UTC)",
    "Timestamp (Local)",
    "Origin",
    "User Trigger",
    "Use Case (Raw)",
    "Source App",
    "Request",
    "Response",
    "Model",
    "Record Type",
]

def resolve_timezone(spec):
    """Return (tzinfo, label). Accepts local, utc, an offset in hours, or an IANA name."""
    key = (spec or "local").strip()
    low = key.lower()
    if low == "local":
        tz = datetime.now().astimezone().tzinfo or timezone.utc
        return tz, local_label(tz)
    if low in ("utc", "z"):
        return timezone.utc, "UTC"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", key):
        hours = float(key)
        if not -12 <= hours <= 14:
            raise ValueError(f"offset {key} is outside the plausible range -12..+14")
        tz = timezone(timedelta(hours=hours))
        return tz, f"UTC{hours:+g}"
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(key)
        return tz, local_label(tz)
    except Exception as exc:
        raise ValueError(
            f"cannot interpret --tz {spec!r}; use local, utc, an offset like +7 / -4.5, "
            f"or an IANA name like Asia/Jakarta ({exc})"
        ) from None


def local_label(tz):
    now = datetime.now(tz)
    name = now.tzname() or ""
    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return f"{name} {offset}".strip()


def to_epoch(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def format_epoch(ts, tz):
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=tz).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return f"<epoch out of range: {ts:g}>"


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def clamp(text, limit):
    if not limit or len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def preview_text(text, limit=140):
    """One line, hard-capped, for compact table previews. Full text lives in the expanded row."""
    if not text:
        return ""
    first_line = text.splitlines()[0]
    truncated = len(text) > len(first_line)
    if len(first_line) > limit:
        first_line = first_line[:limit].rstrip()
        truncated = True
    return first_line + ("\u2026" if truncated else "")


def redact_text(text):
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"[redacted chars={len(text)} sha256={digest}]"


def csv_guard(value, enabled):
    """Neutralise leading characters that spreadsheet apps execute as formulas."""
    if not enabled or not value:
        return value
    if value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def classify_origin(raw):
    low = raw.lower()
    if "cloud" in low:
        return "PrivateCloudCompute"
    if "device" in low:
        return "OnDevice"
    return "Unknown"


def user_trigger(*candidates):
    for candidate in candidates:
        text = as_text(candidate)
        for key, label in USE_CASE_TRIGGERS.items():
            if key in text:
                return label
    for candidate in candidates:
        text = as_text(candidate)
        if "VisualGeneration" in text:
            return "Image Generation"
        if "GenerativeAssistant" in text:
            return "Assistant"
        template = re.search(r'templateID:\s*"([^"]+)"', text)
        if template:
            for key, label in USE_CASE_TRIGGERS.items():
                if key in template.group(1):
                    return label
    for candidate in candidates:
        text = as_text(candidate)
        if text:
            return text
    return "Unknown"


def source_app(client_id, use_case):
    text = as_text(client_id)
    for key, label in CLIENT_APPS.items():
        if key in text:
            return label
    if text:
        return text
    prose = as_text(use_case).lower()
    if "safari" in prose:
        return "Safari"
    if "photo" in prose or "memory" in prose:
        return "Photos"
    return ""


def read_prompt(prompt_text):
    text = as_text(prompt_text).strip()
    if not text:
        return ""
    if text == "<image>":
        return "<image input>"
    if text.startswith("{"):
        head = text.split("<n>", 1)
        try:
            payload = json.loads(head[0])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            parts = [as_text(payload.get("prompt"))]
            original = as_text(payload.get("originalText"))
            if original:
                parts.append(f"[on: {original}]")
            if len(head) > 1:
                parts.append(f">> {head[1]}")
            return " ".join(p for p in parts if p)
    if text.startswith("PromptTemplateInfo("):
        return template_prompt(text)
    return text.replace("<n>", " | ")


def template_prompt(text):
    bindings = re.search(r"variableBindings:\s*\[(.+?)\],\s*locale:", text, re.DOTALL)
    if bindings:
        pairs = re.findall(r'"(\w+)":\s*"((?:[^"\\]|\\.)*)"', bindings.group(1))
        values = {}
        for key, value in pairs:
            cleaned = value.replace("\\n", " ").replace("<n>", " ").strip()
            if cleaned:
                values[key] = cleaned
        for key in ("userPrompt", "prompt", "userContent", "doc", "freeformStoryPromptQuery"):
            if values.get(key):
                return values[key]
    strings = re.findall(r'string:\s*"((?:[^"\\]|\\.){10,})"', text)
    if strings:
        return max(strings, key=len).replace("<n>", " ")
    template = re.search(r'templateID:\s*"([^"]+)"', text)
    if template:
        return f"(template prompt: {template.group(1)})"
    return "(unparsed template prompt)"


def read_response(response_text):
    text = as_text(response_text).strip()
    if not text:
        return ""
    if text == "<tool-call>":
        return "<tool call triggered>"
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("content", "body", "summary"):
                if as_text(payload.get(key)):
                    return as_text(payload[key])
    if text.startswith("<file>"):
        inner = text[len("<file>"):].strip()
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            return inner or "<file generated>"
        return as_text(payload.get("content")) or inner or "<file generated>"
    if "<image>" in text:
        return text.replace("<image>", "").strip() or "<image generated>"
    return text


def attestation_note(nodes):
    """Describe node attestation without dumping the multi-kilobyte base64 bundles."""
    if not nodes:
        return "no nodes"
    validated = sum(1 for n in nodes if as_text(n.get("nodeState")) == "Validated")
    with_bundle = sum(1 for n in nodes if as_text(n.get("attestationBundle")))
    blobs = "".join(as_text(n.get("attestationBundle")) for n in nodes)
    digest = hashlib.sha256(blobs.encode("utf-8", "replace")).hexdigest()[:16] if blobs else "none"
    return (
        f"{len(nodes)} node(s), {validated} validated, {with_bundle} with attestation, "
        f"attestation sha256:{digest}"
    )


def parse_report(data, tz, max_content, redact):
    model_requests = data.get("modelRequests") or []
    pcc_requests = data.get("privateCloudComputeRequests") or []
    warnings = []
    records = []

    if not isinstance(model_requests, list):
        warnings.append("modelRequests is not a list; skipped")
        model_requests = []
    if not isinstance(pcc_requests, list):
        warnings.append("privateCloudComputeRequests is not a list; skipped")
        pcc_requests = []

    for index, entry in enumerate(model_requests, 1):
        if not isinstance(entry, dict):
            warnings.append(f"modelRequests[{index}] is not an object; skipped")
            continue
        ts = to_epoch(entry.get("timestamp"))
        if ts is None:
            warnings.append(f"modelRequests[{index}] has no usable timestamp")
        request = read_prompt(entry.get("prompt"))
        response = read_response(entry.get("response"))
        if redact:
            request, response = redact_text(request), redact_text(response)
        records.append(
            {
                "timestamp": ts,
                "timestamp_utc": format_epoch(ts, timezone.utc),
                "timestamp_local": format_epoch(ts, tz),
                "event_id": as_text(entry.get("identifier")),
                "origin": classify_origin(as_text(entry.get("executionEnvironment"))),
                "execution_environment_raw": as_text(entry.get("executionEnvironment")),
                "user_trigger": user_trigger(
                    entry.get("useCase"), entry.get("prompt"), entry.get("model")
                ),
                "use_case": as_text(entry.get("useCase")),
                "source_app": source_app(entry.get("clientIdentifier"), entry.get("useCase")),
                "request": clamp(request, max_content),
                "response": clamp(response, max_content),
                "model": as_text(entry.get("model")),
                "model_version": as_text(entry.get("modelVersion")),
                "record_type": "Model_Request",
                "detail": "",
            }
        )

    for index, entry in enumerate(pcc_requests, 1):
        if not isinstance(entry, dict):
            warnings.append(f"privateCloudComputeRequests[{index}] is not an object; skipped")
            continue
        ts = to_epoch(entry.get("timestamp"))
        if ts is None:
            warnings.append(f"privateCloudComputeRequests[{index}] has no usable timestamp")
        pipeline = as_text(entry.get("pipelineKind"))
        params = entry.get("pipelineParameters")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                warnings.append(f"privateCloudComputeRequests[{index}] has unparseable pipelineParameters")
                params = {}
        if not isinstance(params, dict):
            params = {}
        nodes = entry.get("nodes")
        nodes = [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []
        adapter = as_text(params.get("adapter"))
        records.append(
            {
                "timestamp": ts,
                "timestamp_utc": format_epoch(ts, timezone.utc),
                "timestamp_local": format_epoch(ts, tz),
                "event_id": as_text(entry.get("requestId")),
                "origin": "PrivateCloudCompute",
                "execution_environment_raw": "PrivateCloudCompute",
                "user_trigger": user_trigger(adapter, pipeline),
                "use_case": pipeline,
                "source_app": f"PCC ({pipeline})" if pipeline else "PCC",
                "request": f"pipeline: {pipeline or 'unknown'}",
                "response": f"adapter: {adapter or 'unknown'}",
                "model": as_text(params.get("model")),
                "model_version": "",
                "record_type": "PCC_Request",
                "detail": attestation_note(nodes),
            }
        )

    records.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"]))
    summary = {
        "records": len(records),
        "model_requests": sum(1 for r in records if r["record_type"] == "Model_Request"),
        "pcc_requests": sum(1 for r in records if r["record_type"] == "PCC_Request"),
        "model_entries": len(model_requests),
        "pcc_entries": len(pcc_requests),
        "on_device": sum(1 for r in records if r["origin"] == "OnDevice"),
        "private_cloud": sum(1 for r in records if r["origin"] == "PrivateCloudCompute"),
        "unknown_origin": sum(1 for r in records if r["origin"] == "Unknown"),
        "warnings": warnings,
    }
    return records, summary


def write_csv(records, path, guard):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(CSV_COLUMNS)
        for record in records:
            writer.writerow(
                [
                    record["timestamp_utc"],
                    record["timestamp_local"],
                    record["origin"],
                    record["user_trigger"],
                    record["use_case"],
                    record["source_app"],
                    csv_guard(record["request"], guard),
                    csv_guard(record["response"], guard),
                    record["model"],
                    record["record_type"],
                ]
            )


def write_json(records, summary, path, source_name, tz_label, generated):
    payload = {
        "source": source_name,
        "generated_utc": generated,
        "local_timezone": tz_label,
        "summary": summary,
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_txt(records, summary, path, source_name, tz_label, generated):
    def rule(char="-"):
        return char * 78

    lines = [
        "Apple Intelligence Report, parsed",
        rule("="),
        f"Source          : {source_name}",
        f"Generated (UTC) : {generated}",
        f"Local timezone  : {tz_label}",
        f"Records         : {summary['records']} "
        f"(model {summary['model_requests']}, PCC {summary['pcc_requests']})",
        f"Origin split    : on-device {summary['on_device']}, "
        f"private cloud {summary['private_cloud']}, unknown {summary['unknown_origin']}",
    ]
    skipped = (summary["model_entries"] - summary["model_requests"]) + (
        summary["pcc_entries"] - summary["pcc_requests"]
    )
    if skipped:
        lines.append(f"Skipped entries : {skipped} (not JSON objects, see warnings)")
    stamped = [r for r in records if r["timestamp"] is not None]
    if stamped:
        lines.append(f"Time range (UTC): {stamped[0]['timestamp_utc']} -> {stamped[-1]['timestamp_utc']}")
    lines.append("")

    for record in records:
        lines.append(rule())
        lines.append(f"[{record['timestamp_utc']} UTC | {record['timestamp_local']} local] "
                     f"{record['record_type']} | {record['origin']}")
        lines.append(f"  Event ID     : {record['event_id']}")
        lines.append(f"  User trigger : {record['user_trigger']}")
        lines.append(f"  Use case     : {record['use_case']}")
        lines.append(f"  Source app   : {record['source_app']}")
        lines.append(f"  Model        : {record['model']} {record['model_version']}".rstrip())
        if record["request"]:
            lines.append(f"  Request      : {record['request']}")
        if record["response"]:
            lines.append(f"  Response     : {record['response']}")
        if record["detail"]:
            lines.append(f"  PCC detail   : {record['detail']}")
        lines.append("")

    lines.append(rule("="))
    lines.append("Counts by user trigger (model requests)")
    for label, count in Counter(
        r["user_trigger"] for r in records if r["record_type"] == "Model_Request"
    ).most_common():
        lines.append(f"  {count:4d}  {label}")
    lines.append("")
    lines.append("Counts by source app (model requests)")
    for label, count in Counter(
        r["source_app"] for r in records if r["record_type"] == "Model_Request" and r["source_app"]
    ).most_common():
        lines.append(f"  {count:4d}  {label}")
    lines.append("")
    lines.append("Counts by model")
    for label, count in Counter(r["model"] for r in records if r["model"]).most_common():
        lines.append(f"  {count:4d}  {label}")
    if summary["warnings"]:
        lines.append("")
        lines.append("Warnings")
        for warning in summary["warnings"]:
            lines.append(f"  {warning}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_md(records, summary, path, source_name, tz_label, generated):
    lines = [
        "# Apple Intelligence Report, parsed",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source | {source_name} |",
        f"| Generated (UTC) | {generated} |",
        f"| Local timezone | {tz_label} |",
        f"| Records | {summary['records']} (model {summary['model_requests']}, PCC {summary['pcc_requests']}) |",
        f"| On-device | {summary['on_device']} |",
        f"| Private cloud | {summary['private_cloud']} |",
        f"| Unknown origin | {summary['unknown_origin']} |",
        "",
    ]
    if summary["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Requests")
    lines.append("")
    for record in records:
        origin_badge = {
            "OnDevice": "ON-DEVICE",
            "PrivateCloudCompute": "CLOUD",
            "Unknown": "UNKNOWN",
        }.get(record["origin"], "UNKNOWN")
        lines.append(f"### {record['timestamp_local']} | {origin_badge} | {record['user_trigger']}")
        lines.append("")
        lines.append(f"- **Origin:** {record['origin']}")
        lines.append(f"- **Source app:** {record['source_app'] or 'n/a'}")
        lines.append(f"- **Use case:** {record['use_case'] or 'n/a'}")
        lines.append(f"- **Model:** {record['model'] or 'n/a'}")
        lines.append(f"- **Record type:** {record['record_type']}")
        if record["request"]:
            lines.append(f"- **Request:** {record['request']}")
        if record["response"]:
            lines.append(f"- **Response:** {record['response']}")
        if record["detail"]:
            lines.append(f"- **PCC detail:** {record['detail']}")
        lines.append("")

    lines.append("## Summary counts")
    lines.append("")
    lines.append("### By user trigger (model requests)")
    lines.append("")
    lines.append("| Count | Trigger |")
    lines.append("|---|---|")
    for label, count in Counter(
        r["user_trigger"] for r in records if r["record_type"] == "Model_Request"
    ).most_common():
        lines.append(f"| {count} | {label} |")
    lines.append("")
    lines.append("### By source app (model requests)")
    lines.append("")
    lines.append("| Count | App |")
    lines.append("|---|---|")
    for label, count in Counter(
        r["source_app"] for r in records if r["record_type"] == "Model_Request" and r["source_app"]
    ).most_common():
        lines.append(f"| {count} | {label} |")
    lines.append("")
    lines.append("### By model")
    lines.append("")
    lines.append("| Count | Model |")
    lines.append("|---|---|")
    for label, count in Counter(r["model"] for r in records if r["model"]).most_common():
        lines.append(f"| {count} | {label} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_html(records, summary, path, source_name, tz_label, generated):
    escape = html_mod.escape

    if records:
        rows = []
        for index, record in enumerate(records):
            badge = ORIGIN_CLASSES.get(record["origin"], "unknown")
            detail_id = f"detail-{index}"
            search_blob = " ".join(
                as_text(record.get(key))
                for key in (
                    "timestamp_utc",
                    "timestamp_local",
                    "origin",
                    "user_trigger",
                    "use_case",
                    "source_app",
                    "request",
                    "response",
                    "model",
                    "model_version",
                    "detail",
                    "event_id",
                    "record_type",
                )
            ).lower()
            request_preview = preview_text(record["request"])
            rows.append(
                f'      <tr class="row" tabindex="0" data-detail="{detail_id}" '
                f'data-origin="{escape(record["origin"])}" data-search="{escape(search_blob)}">'
                f'<td class="mono"><span class="chev" aria-hidden="true">\u25b8</span>{escape(record["timestamp_local"])}</td>'
                f'<td><span class="badge {badge}">{escape(record["origin"])}</span></td>'
                f'<td>{escape(record["source_app"])}</td>'
                f'<td>{escape(record["user_trigger"])}</td>'
                f'<td class="preview">{escape(request_preview)}</td>'
                "</tr>"
            )

            detail_fields = [
                ("Timestamp (UTC)", record["timestamp_utc"]),
                ("Use case", record["use_case"]),
                ("Model", record["model"]),
            ]
            if record["model_version"]:
                detail_fields.append(("Model version", record["model_version"]))
            detail_fields.append(("Record type", record["record_type"]))
            if record["event_id"]:
                detail_fields.append(("Event ID", record["event_id"]))
            if record["detail"]:
                detail_fields.append(("PCC detail", record["detail"]))
            meta_html = "".join(
                f'<div><span class="k">{escape(label)}</span>'
                f'<span class="v mono">{escape(as_text(value)) or "&mdash;"}</span></div>'
                for label, value in detail_fields
            )
            rows.append(
                f'      <tr class="detail-row" id="{detail_id}" hidden><td colspan="5">'
                f'<div class="detail-meta">{meta_html}</div>'
                '<div class="detail-block"><span class="k">Full request</span>'
                f'<pre>{escape(record["request"]) or "(empty)"}</pre></div>'
                '<div class="detail-block"><span class="k">Full response</span>'
                f'<pre>{escape(record["response"]) or "(empty)"}</pre></div>'
                "</td></tr>"
            )
        table = (
            "  <table id=\"records\">\n    <thead><tr>"
            '<th tabindex="0">Timestamp (Local)</th>'
            '<th tabindex="0">Origin</th>'
            '<th tabindex="0">App</th>'
            '<th tabindex="0">Trigger</th>'
            '<th tabindex="0">Request</th>'
            "</tr></thead>\n    <tbody>\n"
            + "\n".join(rows)
            + "\n    </tbody>\n  </table>"
        )
    else:
        table = '  <p class="empty">No requests in this report.</p>'

    warnings_html = ""
    if summary["warnings"]:
        items = "".join(f"<li>{escape(w)}</li>" for w in summary["warnings"])
        warnings_html = f'  <div class="warn"><b>Warnings</b><ul>{items}</ul></div>\n'

    total = summary["records"] or 1
    on_device_pct = round(summary["on_device"] / total * 100)
    cloud_pct = round(summary["private_cloud"] / total * 100)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apple Intelligence Report, parsed</title>
<style>
:root {{
  --bg: #f5f5f7;
  --surface: #fff;
  --text: #1d1d1f;
  --text-secondary: #636366;
  --text-tertiary: #86868b;
  --border: #d2d2d7;
  --border-soft: #e5e5ea;
  --accent: #0071e3;
  --accent-hover: #0077ed;
  --on-device-bg: #e8f5e9;
  --on-device-text: #1b5e20;
  --cloud-bg: #e3f2fd;
  --cloud-text: #0d47a1;
  --unknown-bg: #f5f5f7;
  --unknown-text: #636366;
  --warn-bg: #fff3e0;
  --warn-border: #ff9800;
  --code-bg: #f5f5f7;
  --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #000;
    --surface: #1c1c1e;
    --text: #f5f5f7;
    --text-secondary: #a1a1a6;
    --text-tertiary: #6e6e73;
    --border: #38383a;
    --border-soft: #2c2c2e;
    --accent: #0a84ff;
    --accent-hover: #409cff;
    --on-device-bg: #1b3a28;
    --on-device-text: #a5d6a7;
    --cloud-bg: #102a4e;
    --cloud-text: #90caf9;
    --unknown-bg: #2c2c2e;
    --unknown-text: #a1a1a6;
    --warn-bg: #3d2a00;
    --warn-border: #ff9f0a;
    --code-bg: #2c2c2e;
    --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
  }}
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif;
  background: var(--bg);
  color: var(--text);
  padding: 24px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
.header {{
  background: var(--surface);
  border-radius: 12px;
  padding: 28px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}}
h1 {{
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
}}
h1 span {{ font-weight: 400; color: var(--text-tertiary); font-size: 14px; margin-left: 8px; }}
.sub {{ font-size: 13px; color: var(--text-secondary); margin-top: 6px; }}
.stats-row {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 20px;
}}
.stat {{
  background: var(--bg);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  color: var(--text-secondary);
  border: 1px solid var(--border-soft);
}}
.stat b {{ font-size: 18px; font-weight: 600; color: var(--text); display: block; }}
.breakdown-bar {{
  display: flex;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  margin-top: 16px;
  background: var(--border-soft);
}}
.breakdown-bar .on-device {{ background: #34c759; flex: {on_device_pct}; }}
.breakdown-bar .cloud {{ background: #007aff; flex: {cloud_pct}; }}
.breakdown-labels {{ display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: var(--text-secondary); }}
.breakdown-labels .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
.breakdown-labels .dot.on-device {{ background: #34c759; }}
.breakdown-labels .dot.cloud {{ background: #007aff; }}
.warn {{
  font-size: 13px;
  background: var(--warn-bg);
  border-left: 3px solid var(--warn-border);
  padding: 12px 14px;
  border-radius: 6px;
  margin-top: 16px;
}}
.warn ul {{ margin: 6px 0 0 20px; padding: 0; }}
.warn li {{ margin-bottom: 2px; }}
.panel {{
  background: var(--surface);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}}
.panel h2 {{ font-size: 16px; font-weight: 600; margin-bottom: 14px; letter-spacing: -0.01em; }}
.controls {{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 14px;
}}
.controls label {{ font-size: 13px; color: var(--text-secondary); }}
input[type=search], select {{
  font: inherit;
  font-size: 13px;
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  min-width: 180px;
}}
select {{ cursor: pointer; }}
input:focus-visible, select:focus-visible, th[tabindex]:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}
.count {{ font-size: 12px; color: var(--text-tertiary); margin-left: auto; }}
.table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border-soft); vertical-align: top; }}
th {{
  background: var(--code-bg);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--text-secondary);
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  position: sticky;
  top: 0;
}}
th:hover {{ background: var(--border-soft); }}
.mono {{ font-family: 'SF Mono', ui-monospace, monospace; font-size: 11px; color: var(--text-secondary); }}
tr.row {{ cursor: pointer; }}
tr.row:hover {{ background: var(--code-bg); }}
tr.row.expanded {{ background: var(--code-bg); }}
tr.row:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
.chev {{
  display: inline-block;
  width: 12px;
  margin-right: 6px;
  color: var(--text-tertiary);
  transition: transform 0.15s ease;
}}
tr.row.expanded .chev {{ transform: rotate(90deg); }}
td.preview {{
  max-width: 480px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
tr.detail-row td {{
  background: var(--code-bg);
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  cursor: default;
}}
.detail-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 28px;
  margin-bottom: 14px;
}}
.detail-meta > div {{ min-width: 140px; }}
.detail-meta .k, .detail-block .k {{
  display: block;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--text-tertiary);
  margin-bottom: 2px;
}}
.detail-meta .v {{ font-size: 12px; color: var(--text); }}
.detail-block {{ margin-top: 12px; }}
.detail-block pre {{
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'SF Mono', ui-monospace, monospace;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
  background: var(--surface);
  border: 1px solid var(--border-soft);
  border-radius: 8px;
  padding: 12px;
  margin-top: 6px;
  max-height: 480px;
  overflow: auto;
}}
.pagination {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 14px;
  font-size: 13px;
  color: var(--text-secondary);
}}
.pagination button, .controls button {{
  font: inherit;
  font-size: 13px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
}}
.pagination button:hover:not(:disabled), .controls button:hover:not(:disabled) {{ background: var(--border-soft); }}
.pagination button:disabled, .controls button:disabled {{ opacity: 0.4; cursor: default; }}
.pagination button:focus-visible, .controls button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}}
.badge.on-device {{ background: var(--on-device-bg); color: var(--on-device-text); }}
.badge.cloud {{ background: var(--cloud-bg); color: var(--cloud-text); }}
.badge.unknown {{ background: var(--unknown-bg); color: var(--unknown-text); }}
.empty {{ font-size: 14px; color: var(--text-tertiary); padding: 12px 0; }}
.footer {{
  font-size: 12px;
  color: var(--text-tertiary);
  padding: 8px 4px 0;
}}
@media (max-width: 768px) {{
  body {{ padding: 12px; }}
  .header, .panel {{ padding: 16px; border-radius: 10px; }}
  .stats-row {{ gap: 6px; }}
  .stat {{ padding: 8px 10px; flex: 1 1 calc(50% - 6px); }}
  .stat b {{ font-size: 16px; }}
  .controls {{ flex-direction: column; align-items: stretch; }}
  input[type=search], select {{ min-width: 0; width: 100%; }}
  .count {{ margin-left: 0; text-align: right; }}
  td.preview {{ max-width: 160px; }}
  .detail-meta {{ gap: 8px 16px; }}
  .detail-meta > div {{ min-width: 45%; }}
  th, td {{ padding: 8px; }}
  h1 {{ font-size: 18px; }}
  .pagination {{ flex-wrap: wrap; }}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Apple Intelligence Report <span>parsed</span></h1>
  <div class="sub">Source: {escape(source_name)}</div>
  <div class="sub">Generated {escape(generated)} UTC, local timezone {escape(tz_label)}</div>
  <div class="stats-row">
    <div class="stat">records<b>{summary["records"]}</b></div>
    <div class="stat">model requests<b>{summary["model_requests"]}</b></div>
    <div class="stat">PCC requests<b>{summary["pcc_requests"]}</b></div>
    <div class="stat">on-device<b>{summary["on_device"]}</b></div>
    <div class="stat">private cloud<b>{summary["private_cloud"]}</b></div>
  </div>
  <div class="breakdown-bar">
    <div class="on-device"></div>
    <div class="cloud"></div>
  </div>
  <div class="breakdown-labels">
    <span><span class="dot on-device"></span>On-device {on_device_pct}%</span>
    <span><span class="dot cloud"></span>Private cloud {cloud_pct}%</span>
  </div>
{warnings_html}</div>
<div class="panel">
  <h2>Requests</h2>
  <div class="controls">
    <label for="q">Filter</label>
    <input type="search" id="q" placeholder="text in any column, including full request/response">
    <label for="origin">Origin</label>
    <select id="origin">
      <option value="">all</option>
      <option value="OnDevice">on-device</option>
      <option value="PrivateCloudCompute">private cloud</option>
      <option value="Unknown">unknown</option>
    </select>
    <button type="button" id="expandAll">Expand all</button>
    <button type="button" id="collapseAll">Collapse all</button>
    <span class="count" id="count"></span>
  </div>
  <div class="table-scroll">
{table}
  </div>
  <div class="pagination">
    <button type="button" id="prevPage">Previous</button>
    <span id="pageInfo">Page 1 of 1</span>
    <button type="button" id="nextPage">Next</button>
  </div>
</div>
<div class="footer">Click a row to expand the full request and response. Click a column heading to sort. Filtering and pagination run locally; this file loads nothing from the network.</div>
<script>
const PAGE_SIZE = 25;
const table = document.getElementById('records');
const tbody = table ? table.querySelector('tbody') : null;
let dataRows = tbody ? Array.from(tbody.querySelectorAll('tr.row')) : [];
const search = document.getElementById('q');
const originSelect = document.getElementById('origin');
const count = document.getElementById('count');
const pageInfo = document.getElementById('pageInfo');
const prevBtn = document.getElementById('prevPage');
const nextBtn = document.getElementById('nextPage');
const expandAllBtn = document.getElementById('expandAll');
const collapseAllBtn = document.getElementById('collapseAll');

let filteredRows = dataRows.slice();
let currentPage = 1;

function detailFor(row) {{
  const id = row.getAttribute('data-detail');
  return id ? document.getElementById(id) : null;
}}

function setExpanded(row, expanded) {{
  const detail = detailFor(row);
  if (!detail) return;
  detail.hidden = !expanded;
  row.classList.toggle('expanded', expanded);
}}

function currentPageRows() {{
  const start = (currentPage - 1) * PAGE_SIZE;
  return filteredRows.slice(start, start + PAGE_SIZE);
}}

function render() {{
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;
  const visible = new Set(currentPageRows());
  dataRows.forEach(row => {{
    const show = visible.has(row);
    row.hidden = !show;
    if (!show) setExpanded(row, false);
  }});
  if (count) count.textContent = filteredRows.length + ' of ' + dataRows.length + ' shown';
  if (pageInfo) pageInfo.textContent = 'Page ' + currentPage + ' of ' + totalPages;
  if (prevBtn) prevBtn.disabled = currentPage <= 1;
  if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
}}

function applyFilters() {{
  const needle = (search && search.value || '').toLowerCase();
  const want = originSelect ? originSelect.value : '';
  filteredRows = dataRows.filter(row => {{
    const hay = row.getAttribute('data-search') || '';
    const origin = row.getAttribute('data-origin') || '';
    return (!needle || hay.includes(needle)) && (!want || origin === want);
  }});
  currentPage = 1;
  render();
}}

if (search) search.addEventListener('input', applyFilters);
if (originSelect) originSelect.addEventListener('change', applyFilters);
if (prevBtn) prevBtn.addEventListener('click', () => {{ currentPage -= 1; render(); }});
if (nextBtn) nextBtn.addEventListener('click', () => {{ currentPage += 1; render(); }});
if (expandAllBtn) expandAllBtn.addEventListener('click', () => {{ currentPageRows().forEach(row => setExpanded(row, true)); }});
if (collapseAllBtn) collapseAllBtn.addEventListener('click', () => {{ currentPageRows().forEach(row => setExpanded(row, false)); }});

dataRows.forEach(row => {{
  row.addEventListener('click', () => {{
    const detail = detailFor(row);
    if (!detail) return;
    setExpanded(row, detail.hidden);
  }});
  row.addEventListener('keydown', event => {{
    if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); row.click(); }}
  }});
}});

applyFilters();

let sortState = {{}};
if (table) {{
  const headers = Array.from(table.querySelectorAll('th'));
  headers.forEach((th, index) => {{
    function sort() {{
      sortState[index] = !sortState[index];
      const direction = sortState[index] ? 1 : -1;
      dataRows.sort((a, b) => {{
        const left = (a.cells[index] ? a.cells[index].textContent : '').trim();
        const right = (b.cells[index] ? b.cells[index].textContent : '').trim();
        return left.localeCompare(right, undefined, {{numeric: true}}) * direction;
      }});
      dataRows.forEach(row => {{
        tbody.appendChild(row);
        const detail = detailFor(row);
        if (detail) tbody.appendChild(detail);
      }});
      applyFilters();
    }}
    th.addEventListener('click', sort);
    th.addEventListener('keydown', event => {{
      if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); sort(); }}
    }});
  }});
}}
</script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def synthetic_report():
    return {
        "modelRequests": [
            {
                "timestamp": 1757900000.5,
                "identifier": "11111111-1111-1111-1111-111111111111",
                "useCase": "com.apple.SummarizationKit.mailMessage.synopsis",
                "prompt": (
                    'PromptTemplateInfo(templateID: "com.apple.SummarizationKit.mailMessage.synopsis", '
                    'variableBindings: ["context": "", "doc": "Subject: Demo\\nBody: synthetic body"], '
                    "locale: Optional(en_US))"
                ),
                "response": "Synthetic summary of the demo message.",
                "model": "com.apple.fm.language.instruct_on_device_v1.text_summarizer",
                "modelVersion": "1.0.0",
                "clientIdentifier": "com.apple.mobilemail",
                "executionEnvironment": "OnDevice",
            },
            {
                "timestamp": 1757900100.0,
                "identifier": "22222222-2222-2222-2222-222222222222",
                "useCase": "GenerativeAssistant.knowledge",
                "prompt": "What is the airspeed velocity of an unladen swallow?",
                "response": "Roughly 11 m/s for a European swallow.",
                "model": "com.apple.fm.language.instruct_server_v1.base",
                "modelVersion": "8.0.0",
                "clientIdentifier": "com.apple.siri",
                "executionEnvironment": "PrivateCloudCompute",
            },
        ],
        "privateCloudComputeRequests": [
            {
                "timestamp": 1757900100.1,
                "requestId": "22222222-2222-2222-2222-222222222222",
                "pipelineKind": "demo-pipeline",
                "pipelineParameters": json.dumps(
                    {
                        "adapter": "com.apple.fm.language.instruct_server_v1.base",
                        "model": "com.apple.fm.language.instruct_server_v1.base",
                    }
                ),
                "nodes": [
                    {
                        "node": "DEMONODE0000000000000000000000000000000000000=",
                        "nodeState": "Validated",
                        "attestationBundle": '{"sepAttestation":"MIIFHTCCBKQCAQEwggJPoh8EHTYwMjItc3ludGhldGlj"}',
                    }
                ],
            }
        ],
    }


def check_output_paths(input_path, out_dir, targets, force):
    resolved_out = os.path.realpath(out_dir)
    for prefix in BLOCKED_OUTPUT_PREFIXES:
        if resolved_out == prefix or resolved_out.startswith(prefix + os.sep):
            print(f"refusing to write into {resolved_out}: this looks like an operating system location")
            sys.exit(3)
    if os.path.exists(resolved_out) and not os.path.isdir(resolved_out):
        print(f"refusing to write into {resolved_out}: not a directory")
        sys.exit(3)

    resolved_input = os.path.realpath(input_path)
    for target in targets:
        if os.path.realpath(target) == resolved_input:
            print(f"refusing to overwrite the input report ({target})")
            sys.exit(3)
        if os.path.exists(target) and not force:
            print(f"{target} already exists; pass --force to overwrite")
            sys.exit(2)


def load_report(path):
    if not os.path.isfile(path):
        print(f"input error: {path} is not a regular file")
        sys.exit(1)
    size = os.path.getsize(path)
    if size > MAX_INPUT_BYTES:
        print(f"input error: {path} is {size / 1048576:.0f} MB, above the 512 MB limit")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError:
        print("input error: the file is not UTF-8 text; is this really an AIR JSON export?")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"input error: invalid JSON at line {exc.lineno}, column {exc.colno} ({exc.msg})")
        sys.exit(1)
    except OSError as exc:
        print(f"input error: {exc}")
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse an Apple Intelligence Report JSON into CSV, HTML, TXT and JSON.",
        epilog="Exit codes: 0 ok, 1 input error, 2 output exists, 3 refused output location.",
    )
    parser.add_argument("report", nargs="?", help="path to Apple_Intelligence_Report.json")
    parser.add_argument(
        "--tz",
        default="local",
        help="local-time zone: local (default), utc, an offset in hours (+7, -4.5), or an IANA name",
    )
    parser.add_argument("--out", default="air_parsed", help="output directory (default: ./air_parsed)")
    parser.add_argument(
        "--formats",
        default="csv,html,txt,md,json",
        help="comma-separated subset of csv,html,txt,md,json (default: all five)",
    )
    parser.add_argument("--redact", action="store_true", help="replace request/response text with a hash and length")
    parser.add_argument(
        "--max-content",
        type=int,
        default=2000,
        help="truncate each request/response after N characters, 0 for no limit (default: 2000)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    parser.add_argument("--no-csv-guard", action="store_true", help="do not neutralise spreadsheet formulas")
    parser.add_argument("--demo", action="store_true", help="run on a built-in synthetic report and exit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in ("csv", "html", "txt", "md", "json")]
    if unknown:
        print(f"input error: unknown format(s): {', '.join(unknown)}")
        sys.exit(1)

    if args.demo:
        out_dir = args.out
        input_path = os.path.join(out_dir, "demo_report.json")
    else:
        if not args.report:
            print("input error: give the path to a report, or use --demo")
            sys.exit(1)
        input_path = args.report
        out_dir = args.out

    try:
        tz, tz_label = resolve_timezone(args.tz)
    except ValueError as exc:
        print(f"input error: {exc}")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    targets = {fmt: os.path.join(out_dir, f"{stem}_parsed.{fmt}") for fmt in formats}
    check_output_paths(input_path, out_dir, list(targets.values()), args.force)

    if args.demo:
        os.makedirs(out_dir, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as handle:
            json.dump(synthetic_report(), handle, indent=2)
            handle.write("\n")
        print(f"synthetic report written to {input_path}")

    data = load_report(input_path)
    if not isinstance(data, dict):
        print("input error: the report's top level is not a JSON object")
        sys.exit(1)
    if not data.get("modelRequests") and not data.get("privateCloudComputeRequests"):
        print("note: this report contains no modelRequests and no privateCloudComputeRequests")

    records, summary = parse_report(data, tz, args.max_content, args.redact)
    source_name = os.path.basename(input_path)

    if "csv" in targets:
        write_csv(records, targets["csv"], not args.no_csv_guard)
    if "txt" in targets:
        write_txt(records, summary, targets["txt"], source_name, tz_label, generated)
    if "html" in targets:
        write_html(records, summary, targets["html"], source_name, tz_label, generated)
    if "md" in targets:
        write_md(records, summary, targets["md"], source_name, tz_label, generated)
    if "json" in targets:
        write_json(records, summary, targets["json"], source_name, tz_label, generated)

    print(f"parsed {summary['records']} record(s): {summary['model_requests']} model, "
          f"{summary['pcc_requests']} PCC")
    print(f"origin: {summary['on_device']} on-device, {summary['private_cloud']} private cloud, "
          f"{summary['unknown_origin']} unknown")
    for fmt in formats:
        print(f"  wrote {targets[fmt]}")
    for warning in summary["warnings"]:
        print(f"  warning: {warning}")
    if args.redact:
        print("request and response text were replaced with hashes; metadata is intact")
    else:
        print("these outputs contain your prompts and the model's replies in plain text: keep them private")
    return 0


if __name__ == "__main__":
    sys.exit(main())
