"""Regression tests for air_parse.py.

Stdlib only, matching the tool's own constraint. Run from the repo root:

    python3 -m unittest discover -s tests -v

The first three tests (F1, F2, F6) were written against the behavior reported in
an external eval, so they are expected to FAIL before those fixes land and pass
after. The rest pin behavior that should not change.
"""

import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AIR_PARSE_PATH = REPO_ROOT / "air_parse.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_input.json"
GOLDEN_CSV = REPO_ROOT / "tests" / "golden" / "csv.expected"

sys.path.insert(0, str(REPO_ROOT))
import air_parse  # noqa: E402


def run_cli(args):
    """Invoke main() and return (exit_code, captured_output)."""
    buf = io.StringIO()
    code = 0
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            result = air_parse.main(args)
            code = 0 if result is None else result
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buf.getvalue()


class TestF1MissingOutDir(unittest.TestCase):
    """F1: a real run must not crash when --out does not exist yet."""

    def test_real_run_creates_a_missing_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fresh_output_dir")
            self.assertFalse(os.path.exists(out))
            code, output = run_cli(
                [str(FIXTURE), "--out", out, "--formats", "csv", "--tz", "utc"]
            )
            self.assertEqual(code, 0, f"run failed:\n{output}")
            self.assertTrue(
                os.path.isfile(os.path.join(out, "sample_input_parsed.csv")),
                "CSV was not written into the freshly created directory",
            )

    def test_missing_out_dir_is_not_created_for_blocked_paths(self):
        """The safety gate still wins: a refused location exits 3, never 0."""
        code, _ = run_cli([str(FIXTURE), "--out", "/usr", "--formats", "csv"])
        self.assertEqual(code, 3)

    def test_existing_output_without_force_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = run_cli([str(FIXTURE), "--out", tmp, "--formats", "csv", "--tz", "utc"])
            self.assertEqual(code, 0)
            code2, output2 = run_cli([str(FIXTURE), "--out", tmp, "--formats", "csv", "--tz", "utc"])
            self.assertEqual(code2, 2, output2)


class TestF2TriggerCase(unittest.TestCase):
    """F2: use-case to label matching must ignore case on both sides."""

    def test_mixed_case_identifier_matches(self):
        self.assertEqual(
            air_parse.user_trigger("com.apple.SummarizationKit.fooBar", None, None),
            "Summarize",
        )

    def test_lowercase_identifier_matches(self):
        self.assertEqual(air_parse.user_trigger("summarization", None, None), "Summarize")

    def test_camel_case_key_matches_a_lowercased_identifier(self):
        """The bug is bidirectional: a lowercase spelling of a camelCase key misses today."""
        self.assertEqual(air_parse.user_trigger("writingtools.compose", None, None), "Compose")
        self.assertEqual(
            air_parse.user_trigger("textcomposition.takeawaystransform", None, None),
            "Key Points",
        )

    def test_known_use_case_is_not_contaminated_by_prompt_text(self):
        """F3 correction: a useCase that matches a key always wins over prompt words."""
        self.assertEqual(
            air_parse.user_trigger("GenerativeAssistant.knowledge", "please apply summarization", None),
            "Knowledge",
        )

    def test_unknown_use_case_still_falls_back_to_prompt_text(self):
        """F3 corrected scope: contamination only happens when useCase matches no key."""
        self.assertEqual(
            air_parse.user_trigger("com.apple.BrandNewThing", "please apply summarization", None),
            "Summarize",
        )


class TestF6TemplateBindings(unittest.TestCase):
    """F6: binding extraction must not require a trailing ', locale:'."""

    BINDING_CONTENT = "the real document text here"

    def test_bindings_with_trailing_locale(self):
        text = (
            'PromptTemplateInfo(templateID: "com.apple.X", '
            'variableBindings: [{"userPrompt": "'
            + self.BINDING_CONTENT
            + '"}], locale: "en_US")'
        )
        self.assertEqual(air_parse.template_prompt(text), self.BINDING_CONTENT)

    def test_bindings_without_trailing_locale(self):
        text = (
            'PromptTemplateInfo(templateID: "com.apple.X", '
            'variableBindings: [{"userPrompt": "'
            + self.BINDING_CONTENT
            + '"}])'
        )
        self.assertEqual(
            air_parse.template_prompt(text),
            self.BINDING_CONTENT,
            "binding content was dropped because the regex requires ', locale:'",
        )

    def test_bindings_as_the_last_key_before_a_bare_paren(self):
        text = (
            'PromptTemplateInfo(templateID: "com.apple.Y", '
            'variableBindings: [{"prompt": "' + self.BINDING_CONTENT + '"}])'
        )
        self.assertEqual(air_parse.template_prompt(text), self.BINDING_CONTENT)


class TestF4TimezoneLabel(unittest.TestCase):
    """F4: the printed label belongs to the records, not to the current date."""

    @staticmethod
    def _write(tmp, name, timestamps):
        payload = {
            "modelRequests": [
                {
                    "timestamp": ts,
                    "identifier": f"id-{index}",
                    "useCase": "summarization",
                    "prompt": "hello",
                    "response": "world",
                    "model": "com.apple.fm",
                    "modelVersion": "1.0",
                    "clientIdentifier": "com.apple.mail",
                    "executionEnvironment": "OnDevice",
                }
                for index, ts in enumerate(timestamps)
            ]
        }
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def _labels(self, tmp, timestamps):
        report = self._write(tmp, "report.json", timestamps)
        out = os.path.join(tmp, "out")
        code, output = run_cli([report, "--out", out, "--formats", "txt,json", "--tz", "America/New_York"])
        self.assertEqual(code, 0, output)
        with open(os.path.join(out, "report_parsed.txt"), encoding="utf-8") as fh:
            txt = fh.read()
        with open(os.path.join(out, "report_parsed.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        header = next(line for line in txt.splitlines() if "Local timezone" in line)
        return header, payload["local_timezone"]

    def test_a_report_spanning_dst_names_both_offsets(self):
        january = datetime(2026, 1, 15, 12, tzinfo=timezone.utc).timestamp()
        july = datetime(2026, 7, 15, 12, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            header, json_label = self._labels(tmp, [january, july])
        for label in ("EST -05:00", "EDT -04:00"):
            self.assertIn(label, header, header)
            self.assertIn(label, json_label, json_label)

    def test_a_winter_report_is_labelled_with_the_winter_offset(self):
        january = datetime(2026, 1, 15, 12, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            header, json_label = self._labels(tmp, [january])
        self.assertIn("EST -05:00", header, header)
        self.assertIn("EST -05:00", json_label, json_label)
        self.assertNotIn("EDT", header, header)

    def test_a_fixed_offset_label_is_unchanged(self):
        ts = datetime(2026, 1, 15, 12, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write(tmp, "report.json", [ts])
            out = os.path.join(tmp, "out")
            code, output = run_cli([report, "--out", out, "--formats", "txt", "--tz", "+7"])
            self.assertEqual(code, 0, output)
            with open(os.path.join(out, "report_parsed.txt"), encoding="utf-8") as fh:
                header = next(line for line in fh if "Local timezone" in line)
        self.assertIn("UTC+7", header)


class TestF5MaxContent(unittest.TestCase):
    """F5: a negative --max-content must be rejected, not silently truncate."""

    def test_negative_value_is_a_clean_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            code, output = run_cli(
                [str(FIXTURE), "--out", out, "--formats", "csv", "--tz", "utc", "--max-content", "-1"]
            )
            self.assertEqual(code, 1, f"expected exit 1, got {code}:\n{output}")
            self.assertIn("max-content", output)
            self.assertNotIn("Traceback", output)
            self.assertFalse(
                os.path.exists(os.path.join(out, "sample_input_parsed.csv")),
                "a rejected run must not write output",
            )

    def test_zero_still_means_no_limit(self):
        self.assertEqual(air_parse.clamp("abcdef", 0), "abcdef")

    def test_positive_limit_is_unchanged(self):
        self.assertEqual(air_parse.clamp("abcdef", 3), "abc... [+3 chars]")

    def test_a_positive_limit_truncates_and_reports_the_remainder(self):
        """Cross-check the clamped cell against the uncapped cell from the same report."""
        with tempfile.TemporaryDirectory() as tmp:
            full_out = os.path.join(tmp, "full")
            cap_out = os.path.join(tmp, "cap")
            code, output = run_cli(
                [str(FIXTURE), "--out", full_out, "--formats", "csv", "--tz", "utc", "--max-content", "0"]
            )
            self.assertEqual(code, 0, output)
            code, output = run_cli(
                [str(FIXTURE), "--out", cap_out, "--formats", "csv", "--tz", "utc", "--max-content", "4"]
            )
            self.assertEqual(code, 0, output)
            with open(os.path.join(full_out, "sample_input_parsed.csv"), encoding="utf-8") as fh:
                full = list(__import__("csv").reader(fh))[1][6]
            with open(os.path.join(cap_out, "sample_input_parsed.csv"), encoding="utf-8") as fh:
                capped = list(__import__("csv").reader(fh))[1][6]
            self.assertEqual(capped, f"{full[:4]}... [+{len(full) - 4} chars]")


class TestF7CsvGuardScope(unittest.TestCase):
    """F7: every value the report controls is guarded, not just request/response."""

    HOSTILE = {
        "identifier": "@cmd",
        "useCase": "=SUM(1+1)",
        "prompt": "=1+1",
        "response": "@SUM(2)",
        "model": "-cmd|' /C calc'!A1",
        "clientIdentifier": '=HYPERLINK("http://example.invalid")',
        "executionEnvironment": "OnDevice",
        "timestamp": 1757900000.0,
    }

    @staticmethod
    def _cells(tmp, guard=True):
        report = os.path.join(tmp, "report.json")
        with open(report, "w", encoding="utf-8") as fh:
            json.dump({"modelRequests": [dict(TestF7CsvGuardScope.HOSTILE)]}, fh)
        out = os.path.join(tmp, "out")
        args = [report, "--out", out, "--formats", "csv", "--tz", "utc"]
        if not guard:
            args.append("--no-csv-guard")
        code, output = run_cli(args)
        assert code == 0, output
        with open(os.path.join(out, "report_parsed.csv"), encoding="utf-8") as fh:
            rows = list(__import__("csv").reader(fh))
        return dict(zip(rows[0], rows[1]))

    def test_report_controlled_columns_carry_the_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells = self._cells(tmp)
        for column in ("User Trigger", "Use Case (Raw)", "Source App", "Model"):
            self.assertTrue(cells[column].startswith("'"), f"{column} was not guarded: {cells[column]!r}")
        self.assertTrue(cells["Request"].startswith("'"))
        self.assertTrue(cells["Response"].startswith("'"))

    def test_columns_the_tool_generates_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells = self._cells(tmp)
        self.assertEqual(cells["Origin"], "OnDevice")
        self.assertEqual(cells["Record Type"], "Model_Request")
        self.assertFalse(cells["Timestamp (UTC)"].startswith("'"))

    def test_no_csv_guard_leaves_every_value_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells = self._cells(tmp, guard=False)
        self.assertEqual(cells["Use Case (Raw)"], "=SUM(1+1)")
        self.assertEqual(cells["Request"], "=1+1")

    def test_leading_whitespace_does_not_bypass_the_guard(self):
        for value in (" =SUM(1)", "   +1", "\t@cmd", " \r-2"):
            self.assertTrue(
                air_parse.csv_guard(value, True).startswith("'"),
                f"leading whitespace bypassed the guard for {value!r}",
            )

    def test_ordinary_values_are_untouched(self):
        for value in ("Summarize", "com.apple.mail", "2026-01-15 07:00:00", ""):
            self.assertEqual(air_parse.csv_guard(value, True), value)


class TestWriters(unittest.TestCase):
    """Every format lands, is non-empty, and is self-consistent."""

    def _generate(self, tmp, formats):
        out = os.path.join(tmp, "out")
        code, output = run_cli(
            [str(FIXTURE), "--out", out, "--formats", formats, "--tz", "utc"]
        )
        self.assertEqual(code, 0, output)
        return out

    def test_all_five_formats_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._generate(tmp, "csv,html,txt,md,json")
            for ext in ("csv", "html", "txt", "md", "json"):
                path = os.path.join(out, f"sample_input_parsed.{ext}")
                self.assertTrue(os.path.isfile(path), f"{ext} missing")
                self.assertGreater(os.path.getsize(path), 0, f"{ext} is empty")

    def test_json_output_is_parseable_and_keeps_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._generate(tmp, "json")
            with open(os.path.join(out, "sample_input_parsed.json"), encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["summary"]["records"], 1)
            record = payload["records"][0]
            self.assertEqual(record["origin"], "OnDevice")
            self.assertEqual(record["record_type"], "Model_Request")

    def test_csv_header_and_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._generate(tmp, "csv")
            with open(os.path.join(out, "sample_input_parsed.csv"), encoding="utf-8") as fh:
                rows = list(__import__("csv").reader(fh))
            self.assertEqual(rows[0], air_parse.CSV_COLUMNS)
            self.assertEqual(len(rows), 2, "expected a header plus exactly one record")

    def test_csv_output_is_byte_stable(self):
        """Golden-file check: csv has no generated timestamp, so it must not drift."""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._generate(tmp, "csv")
            produced = pathlib.Path(out, "sample_input_parsed.csv").read_bytes()
        self.assertTrue(GOLDEN_CSV.is_file(), f"missing golden file: {GOLDEN_CSV}")
        self.assertEqual(produced, GOLDEN_CSV.read_bytes(), "CSV output changed")

    def test_choose_a_subset_of_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._generate(tmp, "csv,md")
            present = sorted(os.listdir(out))
            self.assertEqual(present, ["sample_input_parsed.csv", "sample_input_parsed.md"])

    def test_redact_hides_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            code, _ = run_cli([str(FIXTURE), "--out", out, "--formats", "json", "--tz", "utc", "--redact"])
            self.assertEqual(code, 0)
            text = pathlib.Path(out, "sample_input_parsed.json").read_text(encoding="utf-8")
            self.assertNotIn("PLAINTEXT_PROMPT_MARKER", text)
            self.assertIn("[redacted chars=", text)


class TestSecurityContract(unittest.TestCase):
    """The tool's headline promise: stdlib only, no network, no code execution."""

    # These are searched for as ABSENT literals; the test fails if the tool ever
    # contains them. No eval/exec is invoked here or anywhere in this file.
    FORBIDDEN = (
        "import socket",
        "import subprocess",
        "import urllib",
        "from urllib",
        "import requests",
        "os.system",
        "popen(",
        "eval(",
        "exec(",
    )

    def test_source_has_no_network_or_execution_primitives(self):
        source = AIR_PARSE_PATH.read_text(encoding="utf-8")
        for needle in self.FORBIDDEN:
            self.assertNotIn(needle, source, f"found forbidden primitive: {needle!r}")

    def test_only_stdlib_modules_are_imported(self):
        source = AIR_PARSE_PATH.read_text(encoding="utf-8")
        allowed = {
            "__future__", "argparse", "csv", "hashlib", "html", "json", "os", "re",
            "sys", "collections", "datetime", "zoneinfo",
        }
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                name = stripped[len("import "):].split(".")[0].split(" ")[0]
            elif stripped.startswith("from ") and " import " in stripped:
                name = stripped[len("from "):].split(" import ")[0].split(".")[0]
            else:
                continue
            self.assertIn(name, allowed, f"non-stdlib import: {name}")


if __name__ == "__main__":
    unittest.main()
