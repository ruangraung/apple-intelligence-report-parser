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
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
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


class TestF8BreakdownBar(unittest.TestCase):
    """F8: the bar accounts for every record and does not round its segments."""

    @staticmethod
    def _html(tmp, records):
        report = os.path.join(tmp, "report.json")
        with open(report, "w", encoding="utf-8") as fh:
            json.dump({"modelRequests": records}, fh)
        out = os.path.join(tmp, "out")
        code, output = run_cli([report, "--out", out, "--formats", "html", "--tz", "utc"])
        assert code == 0, output
        with open(os.path.join(out, "report_parsed.html"), encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _entry(index, environment):
        return {
            "timestamp": 1757900000.0 + index * 60,
            "identifier": f"id-{index}",
            "useCase": "summarization",
            "prompt": "hello",
            "response": "world",
            "model": "com.apple.fm",
            "modelVersion": "1.0",
            "clientIdentifier": "com.apple.mail",
            "executionEnvironment": environment,
        }

    def test_an_unknown_record_gets_its_own_segment_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._html(tmp, [
                self._entry(0, "OnDevice"),
                self._entry(1, "PrivateCloudCompute"),
                self._entry(2, "SomethingElse"),
            ])
        for segment in ("on-device", "cloud", "unknown"):
            self.assertRegex(
                page,
                rf"\.breakdown-bar \.{segment} \{{[^}}]*flex: 1;",
                f"missing a {segment} segment with an exact weight",
            )
        labels_html = re.search(r'<div class="breakdown-labels">(.*?)</div>\s*<', page, re.S)
        assert labels_html is not None, "no breakdown labels block in the page"
        labels = labels_html.group(1)
        self.assertIn("On-device 33%", labels)
        self.assertIn("Private cloud 33%", labels)
        self.assertIn("Unknown 33%", labels)

    def test_segment_weights_are_record_counts_not_rounded_percentages(self):
        records = [self._entry(i, "OnDevice") for i in range(7)]
        records += [self._entry(10 + i, "PrivateCloudCompute") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            page = self._html(tmp, records)
        self.assertRegex(page, r"\.breakdown-bar \.on-device \{[^}]*flex: 7;")
        self.assertRegex(page, r"\.breakdown-bar \.cloud \{[^}]*flex: 3;")
        self.assertNotRegex(page, r"\.breakdown-bar \.unknown \{")

    def test_an_empty_report_has_no_breakdown_bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._html(tmp, [])
        self.assertNotIn('class="breakdown-bar"', page)
        self.assertNotIn('class="breakdown-labels"', page)


class TestF9ExpandScope(unittest.TestCase):
    """F9: the expand controls are page-scoped, so their labels say so."""

    @staticmethod
    def _page(tmp, count):
        records = [
            {
                "timestamp": 1757900000.0 + index * 60,
                "identifier": f"id-{index}",
                "useCase": "summarization",
                "prompt": "hello",
                "response": "world",
                "model": "com.apple.fm",
                "modelVersion": "1.0",
                "clientIdentifier": "com.apple.mail",
                "executionEnvironment": "OnDevice",
            }
            for index in range(count)
        ]
        report = os.path.join(tmp, "report.json")
        with open(report, "w", encoding="utf-8") as fh:
            json.dump({"modelRequests": records}, fh)
        out = os.path.join(tmp, "out")
        code, output = run_cli([report, "--out", out, "--formats", "html", "--tz", "utc"])
        assert code == 0, output
        with open(os.path.join(out, "report_parsed.html"), encoding="utf-8") as fh:
            return fh.read()

    def test_the_buttons_promise_only_the_visible_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page(tmp, 60)
        self.assertIn(">Expand page<", page)
        self.assertIn(">Collapse page<", page)
        self.assertNotIn(">Expand all<", page)
        self.assertNotIn(">Collapse all<", page)

    def test_the_handler_still_targets_the_current_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page(tmp, 60)
        self.assertIn("currentPageRows().forEach", page)


class TestPCCBranch(unittest.TestCase):
    """The private cloud path, its attestation note, and the warning path.

    The sample fixture ships an empty privateCloudComputeRequests list, so none of
    this code ran in a test before. mixed_input.json carries a model request, two
    private cloud requests and three malformed shapes on purpose.
    """

    MIXED = REPO_ROOT / "tests" / "fixtures" / "mixed_input.json"

    def _payload(self, tmp, formats="json", report=None):
        out = os.path.join(tmp, "out")
        args = [str(report or self.MIXED), "--out", out, "--formats", formats, "--tz", "utc"]
        code, output = run_cli(args)
        self.assertEqual(code, 0, output)
        name = pathlib.Path(report or self.MIXED).stem
        return json.loads(pathlib.Path(out, f"{name}_parsed.json").read_text(encoding="utf-8"))

    def test_both_branches_produce_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(tmp)
        kinds = [record["record_type"] for record in payload["records"]]
        self.assertEqual(kinds.count("Model_Request"), 1)
        self.assertEqual(kinds.count("PCC_Request"), 2)
        self.assertEqual(payload["summary"]["model_requests"], 1)
        self.assertEqual(payload["summary"]["pcc_requests"], 2)

    def test_a_private_cloud_record_reads_pipeline_and_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(tmp)
        record = next(
            item for item in payload["records"]
            if item["record_type"] == "PCC_Request" and item["event_id"].startswith("bbbb")
        )
        self.assertEqual(record["origin"], "PrivateCloudCompute")
        self.assertEqual(record["use_case"], "mixed-pipeline")
        self.assertEqual(record["source_app"], "PCC (mixed-pipeline)")
        self.assertEqual(record["request"], "pipeline: mixed-pipeline")
        self.assertEqual(record["response"], "adapter: com.apple.fm.language.instruct_server_v1.base")
        self.assertEqual(record["model"], "com.apple.fm.language.instruct_server_v1.base")
        self.assertIn("2 node(s)", record["detail"])
        self.assertIn("1 validated", record["detail"])
        self.assertIn("1 with attestation", record["detail"])
        self.assertRegex(record["detail"], r"attestation sha256:[0-9a-f]{16}")

    def test_the_attestation_bundle_never_reaches_an_output(self):
        """The note exists so the multi-kilobyte bundles stay out of every format."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            code, output = run_cli(
                [str(self.MIXED), "--out", out, "--formats", "csv,html,txt,md,json", "--tz", "utc"]
            )
            self.assertEqual(code, 0, output)
            for name in sorted(os.listdir(out)):
                text = pathlib.Path(out, name).read_text(encoding="utf-8")
                self.assertNotIn("BUNDLE_ALPHA_MARKER", text, f"{name} dumped the attestation bundle")

    def test_malformed_entries_become_warnings_not_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(tmp)
        warnings = payload["summary"]["warnings"]
        self.assertIn("modelRequests[2] is not an object; skipped", warnings)
        self.assertIn("privateCloudComputeRequests[2] has no usable timestamp", warnings)
        self.assertIn("privateCloudComputeRequests[2] has unparseable pipelineParameters", warnings)
        self.assertEqual(len(warnings), 3, warnings)

    def test_a_section_that_is_not_a_list_is_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "report.json")
            pathlib.Path(report).write_text(
                json.dumps({"modelRequests": {"oops": 1}, "privateCloudComputeRequests": 7}),
                encoding="utf-8",
            )
            payload = self._payload(tmp, report=report)
        self.assertEqual(payload["summary"]["records"], 0)
        self.assertIn("modelRequests is not a list; skipped", payload["summary"]["warnings"])
        self.assertIn("privateCloudComputeRequests is not a list; skipped", payload["summary"]["warnings"])

    def test_redaction_leaves_the_pipeline_metadata_alone(self):
        """--redact covers user text. A pipeline kind and an adapter name are not that."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            code, output = run_cli(
                [str(self.MIXED), "--out", out, "--formats", "json", "--tz", "utc", "--redact"]
            )
            self.assertEqual(code, 0, output)
            payload = json.loads(
                pathlib.Path(out, "mixed_input_parsed.json").read_text(encoding="utf-8")
            )
        pcc = next(r for r in payload["records"] if r["record_type"] == "PCC_Request")
        model = next(r for r in payload["records"] if r["record_type"] == "Model_Request")
        self.assertEqual(pcc["request"], "pipeline: mixed-pipeline")
        self.assertTrue(model["request"].startswith("[redacted chars="))


class TestDemoGoldenRun(unittest.TestCase):
    """--demo is the path a first-time user takes, so its output is pinned too."""

    GOLDEN = REPO_ROOT / "tests" / "golden" / "demo_report.csv.expected"

    def test_demo_csv_matches_the_golden_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "demo")
            # --tz utc keeps this independent of the machine's local timezone.
            code, output = run_cli(["--demo", "--out", out, "--formats", "csv", "--force", "--tz", "utc"])
            self.assertEqual(code, 0, output)
            produced = pathlib.Path(out, "demo_report_parsed.csv").read_bytes()
        self.assertTrue(self.GOLDEN.is_file(), f"missing golden file: {self.GOLDEN}")
        self.assertEqual(produced, self.GOLDEN.read_bytes(), "demo CSV changed")

    def test_demo_walks_the_private_cloud_path_with_no_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "demo")
            code, output = run_cli(["--demo", "--out", out, "--formats", "json", "--force", "--tz", "utc"])
            self.assertEqual(code, 0, output)
            payload = json.loads(
                pathlib.Path(out, "demo_report_parsed.json").read_text(encoding="utf-8")
            )
        kinds = [record["record_type"] for record in payload["records"]]
        self.assertIn("PCC_Request", kinds)
        self.assertIn("Model_Request", kinds)
        self.assertEqual(payload["summary"]["warnings"], [])


class TestErrorContract(unittest.TestCase):
    """Failures are typed exceptions, and main is the only place that picks a code.

    Before this, load_report and check_output_paths called sys.exit directly, so a
    caller could not tell the failure modes apart without catching SystemExit.
    """

    def test_load_report_reports_a_missing_file_as_an_input_error(self):
        missing = os.path.join(tempfile.gettempdir(), "definitely-not-here.json")
        with self.assertRaises(air_parse.InputError) as caught:
            air_parse.load_report(missing)
        self.assertIn("is not a regular file", str(caught.exception))

    def test_load_report_reports_bad_json_as_an_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            pathlib.Path(path).write_text("{not json", encoding="utf-8")
            with self.assertRaises(air_parse.InputError) as caught:
                air_parse.load_report(path)
        self.assertIn("invalid JSON at line 1", str(caught.exception))

    def test_check_output_paths_refuses_an_os_path(self):
        with self.assertRaises(air_parse.RefusedLocation) as caught:
            air_parse.check_output_paths(str(FIXTURE), "/usr", ["/usr/x_parsed.csv"], False)
        self.assertIn("operating system location", str(caught.exception))

    def test_check_output_paths_refuses_an_existing_target_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "report_parsed.csv")
            pathlib.Path(target).write_text("already here", encoding="utf-8")
            with self.assertRaises(air_parse.OutputExists) as caught:
                air_parse.check_output_paths(str(FIXTURE), tmp, [target], False)
            self.assertIn("pass --force to overwrite", str(caught.exception))
            # --force is the difference between refusing and proceeding.
            air_parse.check_output_paths(str(FIXTURE), tmp, [target], True)

    def test_main_returns_the_code_instead_of_raising_system_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                code = air_parse.main([os.path.join(tmp, "missing.json"), "--out", out])
        self.assertEqual(code, 1)
        self.assertIn("is not a regular file", buf.getvalue())

    def test_the_code_table_matches_the_readme(self):
        table = {kind.__name__: code for kind, code in air_parse.EXIT_CODES}
        self.assertEqual(table, {"InputError": 1, "OutputExists": 2, "RefusedLocation": 3})


class TestHtmlSnapshot(unittest.TestCase):
    """Pin the whole page, not substrings of the source.

    The HTML is what a user actually opens, and until now the tests only checked that
    certain strings appeared in it. The generated-at stamp is passed in, so the
    snapshot does not encode the clock. mixed_input renders both record types.
    """

    GOLDEN = REPO_ROOT / "tests" / "golden" / "mixed_input.html.expected"
    FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mixed_input.json"

    def _render(self, tmp, generated="2026-01-01 00:00:00"):
        data = air_parse.load_report(str(self.FIXTURE_PATH))
        tz, tz_label = air_parse.resolve_timezone("utc")
        records, summary = air_parse.parse_report(data, tz, 2000, False)
        path = os.path.join(tmp, "mixed_input_parsed.html")
        air_parse.write_html(
            records, summary, path, self.FIXTURE_PATH.name, tz_label, generated
        )
        return pathlib.Path(path).read_bytes()

    def test_the_rendered_page_matches_the_golden_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            produced = self._render(tmp)
        self.assertTrue(self.GOLDEN.is_file(), f"missing golden file: {self.GOLDEN}")
        self.assertEqual(produced, self.GOLDEN.read_bytes(), "the HTML output changed")

    def test_the_generated_stamp_is_the_only_clock_dependency(self):
        """If anything else reads the clock, this fails on the line that changed."""
        with tempfile.TemporaryDirectory() as tmp:
            first = self._render(tmp, generated="2026-01-01 00:00:00")
            second = self._render(tmp, generated="2030-12-31 23:59:59")
        first_lines = first.splitlines()
        second_lines = second.splitlines()
        self.assertEqual(len(first_lines), len(second_lines), "the stamp changed the page shape")
        changed = [line for line in second_lines if line not in first_lines]
        self.assertEqual(len(changed), 1, f"more than the stamp moved: {changed}")
        self.assertIn(b"Generated 2030-12-31 23:59:59", changed[0])


class TestRecord(unittest.TestCase):
    """The row shape, and the single place where the text policies live."""

    def _record(self, **overrides):
        base = air_parse.Record(
            timestamp=1735689600.0,
            timestamp_utc="2025-01-01 00:00:00",
            timestamp_local="2025-01-01 07:00:00",
            event_id="abc",
            origin="OnDevice",
            execution_environment_raw="OnDevice",
            user_trigger="Summarise",
            use_case="com.apple.SummarizationKit.mailMessage.synopsis",
            source_app="Mail",
            request="hello",
            response="world",
            model="com.apple.fm.language.instruct_on_device_v1",
            model_version="1.0",
            record_type="Model_Request",
            detail="",
        )
        return replace(base, **overrides) if overrides else base

    def test_parse_report_returns_records(self):
        data = air_parse.load_report(str(FIXTURE))
        tz, _label = air_parse.resolve_timezone("utc")
        records, _summary = air_parse.parse_report(data, tz, 2000, False)
        self.assertTrue(all(isinstance(record, air_parse.Record) for record in records))

    def test_the_json_keys_keep_the_order_writers_relied_on(self):
        self.assertEqual(
            list(self._record().to_json_dict()),
            [
                "timestamp", "timestamp_utc", "timestamp_local", "event_id", "origin",
                "execution_environment_raw", "user_trigger", "use_case", "source_app",
                "request", "response", "model", "model_version", "record_type", "detail",
            ],
        )

    def test_present_with_no_policies_returns_the_same_text(self):
        record = self._record()
        self.assertEqual(record.presented(0, False), record)

    def test_present_clamps_the_request_and_the_response(self):
        record = self._record(request="hello", response="world")
        presented = record.presented(4, False)
        self.assertEqual(presented.request, "hell... [+1 chars]")
        self.assertEqual(presented.response, "worl... [+1 chars]")
        self.assertEqual(presented.event_id, record.event_id, "only the two text fields move")

    def test_present_redacts_before_it_clamps(self):
        record = self._record(request="secret prompt", response="secret answer")
        presented = record.presented(0, True)
        self.assertRegex(presented.request, r"^\[redacted chars=13 sha256=[0-9a-f]{12}\]$")
        self.assertRegex(presented.response, r"^\[redacted chars=13 sha256=[0-9a-f]{12}\]$")
        self.assertNotIn("secret", presented.request + presented.response)

    def test_a_tight_limit_wins_even_over_the_redaction_marker(self):
        """Redaction runs first, so a small --max-content truncates the marker itself."""
        presented = self._record(request="secret prompt").presented(5, True)
        self.assertEqual(presented.request, "[reda... [+34 chars]")


class TestHtmlRenderers(unittest.TestCase):
    """The two renderers are pure functions now, so they can be checked without files."""

    def test_rows_for_an_empty_report(self):
        self.assertEqual(
            air_parse.render_rows([]),
            '  <p class="empty">No requests in this report.</p>',
        )

    def test_rows_carry_the_id_the_toggle_looks_for(self):
        data = air_parse.load_report(str(FIXTURE))
        tz, _label = air_parse.resolve_timezone("utc")
        records, _summary = air_parse.parse_report(data, tz, 2000, False)
        markup = air_parse.render_rows(records)
        self.assertIn('data-detail="detail-0"', markup)
        self.assertIn('id="detail-0" hidden', markup)

    def test_breakdown_is_empty_without_records(self):
        summary = {"records": 0, "on_device": 0, "private_cloud": 0, "unknown_origin": 0}
        self.assertEqual(air_parse.render_breakdown(summary), ("", ""))

    def test_breakdown_labels_follow_the_record_counts(self):
        summary = {"records": 10, "on_device": 7, "private_cloud": 3, "unknown_origin": 0}
        markup, css = air_parse.render_breakdown(summary)
        self.assertIn("On-device 70%", markup)
        self.assertIn("Private cloud 30%", markup)
        self.assertNotIn("Unknown", markup)
        self.assertEqual(css, "", "no unknown records, so no extra CSS")

    def test_breakdown_gives_unknown_records_a_segment_and_its_css(self):
        summary = {"records": 3, "on_device": 1, "private_cloud": 1, "unknown_origin": 1}
        markup, css = air_parse.render_breakdown(summary)
        self.assertIn('class="unknown"', markup)
        self.assertIn("Unknown 33%", markup)
        self.assertIn("flex: 1", css)

    def test_unknown_css_lands_in_the_style_block(self):
        """render_breakdown hands CSS back to write_html, which feeds it to the template."""
        extra = ".breakdown-bar .unknown { flex: 1; }\n"
        css = air_parse.HTML_CSS.substitute(on_device=1, cloud=1, unknown_css=extra)
        self.assertIn(extra, css)
        self.assertNotIn("$", css)

    def test_the_style_and_script_constants_carry_no_escaped_braces(self):
        """They lived in an f-string before, where every literal brace was doubled."""
        self.assertNotIn("{{", air_parse.HTML_CSS.template)
        self.assertNotIn("{{", air_parse.HTML_JS)


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
            "sys", "collections", "datetime", "zoneinfo", "string", "dataclasses",
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
