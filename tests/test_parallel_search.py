import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

from parallel_search import MCP_URL, USER_AGENT, ParallelSearch, ParallelSearchError
from tests.search_workflow_fixture import agents, make_lab, workflow


class Response(io.BytesIO):
    def __init__(self, value, content_type="application/json", raw=False):
        super().__init__(value if raw else json.dumps(value).encode())
        self.headers = {"Content-Type": content_type, "Mcp-Session-Id": "transport-session"}


class RecordingMCP:
    def __init__(self, sse=False, failure=None, content=None):
        self.requests = []
        self.sse = sse
        self.failure = failure
        self.content = content

    def __call__(self, request, timeout):
        message = json.loads(request.data)
        self.requests.append((request, timeout, message))
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif message["method"] == "notifications/initialized":
            return Response(b"", raw=True)
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if self.failure:
                raise self.failure
            name = message["params"]["name"]
            text = ("Research source https://example.org/dataset: benchmark dataset" if name == "web_search"
                    else "Dataset documentation: 500 evaluation examples")
            if self.content is not None:
                text = self.content
            result = {"content": [{"type": "text", "text": text}]}
        value = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        if self.sse:
            raw = b": keepalive\n\ndata: " + json.dumps(value).encode() + b"\n\n"
            return Response(raw, "text/event-stream", raw=True)
        return Response(value)

    @property
    def calls(self):
        return [m["params"] for _, _, m in self.requests if m["method"] == "tools/call"]


def run_commands(lab, commands, transport):
    # Script the model boundary, leaving BaseAgent.inference, prompt/history and
    # LaboratoryWorkflow.literature_review unchanged.
    model = Mock(side_effect=commands + ["```DIALOGUE\nDone\n```"])
    with patch.object(agents, "query_model", model), patch("parallel_search.urlopen", transport), redirect_stdout(io.StringIO()):
        lab.max_steps = len(commands)
        lab.literature_review()
    return model


class WorkflowTests(unittest.TestCase):
    def test_quoted_expiration_in_web_content_is_not_a_history_control(self):
        quoted = "Documentation sample: ```EXPIRATION 3\nquoted content\n```"
        lab, transport = make_lab(True), RecordingMCP(content=quoted)
        model = run_commands(lab, ["```SEARCH_WEB\nresearch docs\n```",
                                   "```FETCH_WEB\nhttps://example.org/docs\n```"], transport)
        self.assertIn(quoted, model.call_args_list[-1].kwargs["prompt"])
        self.assertTrue(all(expiration is None for expiration, _ in lab.phd.history))

    def test_leading_arxiv_expiration_wrapper_still_expires(self):
        lab = make_lab(True)
        arxiv = Mock()
        arxiv.retrieve_full_paper_text.return_value = "paper full text"
        with patch.object(workflow, "ArxivSearch", return_value=arxiv):
            run_commands(lab, ["```FULL_TEXT\n2501.04227\n```"], Mock())
        self.assertEqual(lab.phd.history[-1][0], lab.arxiv_paper_exp_time - 1)
        self.assertIn("paper full text", lab.phd.history[-1][1])

    def test_search_fetch_followup_consumed_by_agent_json_and_sse(self):
        for sse in (False, True):
            with self.subTest(sse=sse):
                lab, transport = make_lab(True), RecordingMCP(sse=sse)
                model = run_commands(lab, ["```SEARCH_WEB\nreasoning evaluation datasets\n```",
                                           "```FETCH_WEB\nhttps://example.org/dataset\n```",
                                           "```SEARCH_WEB\nbenchmark scoring documentation\n```"], transport)
                calls = transport.calls
                self.assertEqual([c["name"] for c in calls], ["web_search", "web_fetch", "web_search"])
                self.assertEqual({c["arguments"]["session_id"] for c in calls}, {lab.web_search_session_id})
                self.assertEqual(calls[0]["arguments"]["search_queries"], ["reasoning evaluation datasets"])
                self.assertEqual(calls[1]["arguments"]["urls"], ["https://example.org/dataset"])
                self.assertIn("SEARCH_WEB", model.call_args_list[0].kwargs["system_prompt"])
                self.assertIn("https://example.org/dataset", model.call_args_list[1].kwargs["prompt"])
                self.assertIn("500 evaluation examples", model.call_args_list[2].kwargs["prompt"])
                self.assertIn("500 evaluation examples", str(lab.phd.history))
                for req, timeout, msg in transport.requests:
                    self.assertEqual(req.full_url, MCP_URL)
                    self.assertEqual(req.get_header("User-agent"), USER_AGENT)
                    self.assertIsNone(req.get_header("Authorization"))
                    self.assertEqual(timeout, 30)
                    if msg["method"] != "initialize":
                        self.assertEqual(req.get_header("Mcp-session-id"), "transport-session")

    def test_default_disabled_and_historical_saved_state_cannot_call(self):
        for historical in (False, True):
            lab = make_lab()
            if historical:
                del lab.parallel_web_search
                del lab.phd.web_search_enabled
            transport = Mock(side_effect=AssertionError("network must be disabled"))
            model = run_commands(lab, ["```SEARCH_WEB\nquery\n```", "```FETCH_WEB\nhttps://example.org\n```"], transport)
            transport.assert_not_called()
            self.assertNotIn("SEARCH_WEB", model.call_args_list[0].kwargs["system_prompt"])
            self.assertIn("Web search is disabled", model.call_args_list[1].kwargs["prompt"])

    def test_arxiv_commands_and_agentrxiv_are_preserved(self):
        for enabled in (False, True):
            lab = make_lab(enabled)
            arxiv = Mock()
            arxiv.find_papers_by_str.return_value = "arxiv summaries"
            arxiv.retrieve_full_paper_text.return_value = "arxiv full text"
            agentrxiv = Mock()
            agentrxiv.num_papers.return_value = 1
            agentrxiv.search_agentrxiv.return_value = "agentrxiv summaries"
            lab.agentRxiv = True
            with patch.object(workflow, "ArxivSearch", return_value=arxiv), patch.object(workflow, "GLOBAL_AGENTRXIV", agentrxiv):
                model = run_commands(lab, ["```SUMMARY\nreasoning\n```", "```FULL_TEXT\n2501.04227\n```",
                                           "```ADD_PAPER\n2501.04227\nRelevant paper\n```"], Mock())
            arxiv.find_papers_by_str.assert_called_once_with("reasoning", N=5)
            arxiv.retrieve_full_paper_text.assert_called_with("2501.04227")
            self.assertEqual(lab.phd.lit_review[0]["arxiv_id"], "2501.04227")
            self.assertIn("agentrxiv summaries", model.call_args_list[1].kwargs["prompt"])

    def test_run_identity_interleaving_shared_client_saved_state_and_reset(self):
        first, second, new_run = make_lab(True), make_lab(True), make_lab(True)
        first.web_search_client = second.web_search_client = ParallelSearch()
        first_id, second_id = first.web_search_session_id, second.web_search_session_id
        self.assertEqual(len({first_id, second_id, new_run.web_search_session_id}), 3)
        transport = RecordingMCP()
        for lab in (first, second, first, second):
            run_commands(lab, ["```SEARCH_WEB\nresearch datasets\n```", "```FETCH_WEB\nhttps://example.org\n```"], transport)
        self.assertEqual([c["arguments"]["session_id"] for c in transport.calls],
                         [first_id, first_id, second_id, second_id] * 2)
        first.reset_agents()  # Phase reset is still part of this research run.
        restored = pickle.loads(pickle.dumps(first))
        run_commands(restored, ["```SEARCH_WEB\nfollow up datasets\n```"], transport)
        self.assertEqual(transport.calls[-1]["arguments"]["session_id"], first_id)

    def test_errors_timeouts_quota_and_retry_feed_back_without_fallback(self):
        errors = [TimeoutError(), URLError("network"),
                  HTTPError(MCP_URL, 429, "quota", {"Retry-After": "60"}, None),
                  HTTPError(MCP_URL, 503, "unavailable", {}, None)]
        for error in errors:
            lab, transport = make_lab(True), RecordingMCP(failure=error)
            arxiv = Mock()
            with patch.object(workflow, "ArxivSearch", return_value=arxiv):
                model = run_commands(lab, ["```SEARCH_WEB\nresearch datasets\n```", "```SEARCH_WEB\nresearch datasets\n```"], transport)
            arxiv.find_papers_by_str.assert_not_called()
            self.assertIn("Web command failed", model.call_args_list[1].kwargs["prompt"])
            self.assertEqual({c["arguments"]["session_id"] for c in transport.calls}, {lab.web_search_session_id})

    def test_keyboard_interrupt_cancels_workflow(self):
        lab, transport = make_lab(True), RecordingMCP(failure=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            run_commands(lab, ["```SEARCH_WEB\nquery\n```"], transport)
        self.assertEqual(len(transport.calls), 1)

    def test_invalid_fetch_urls_are_feedback_without_network(self):
        for url in ("file:///etc/passwd", "https://user:secret@example.org", "http://", "https://example.org/a b", "https://[invalid"):
            lab, transport = make_lab(True), Mock()
            model = run_commands(lab, [f"```FETCH_WEB\n{url}\n```"], transport)
            transport.assert_not_called()
            self.assertIn("requires one HTTP or HTTPS URL", model.call_args_list[1].kwargs["prompt"])

    def test_yaml_default_boolean_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "config.yaml"
            for text, expected in (("research-topic: topic", False), ("parallel-web-search: true", True),
                                   ("parallel-web-search: false", False)):
                file.write_text(text)
                self.assertIs(workflow.parse_yaml(file).parallel_web_search, expected)
            for value in ('"false"', '"true"', "null", "1"):
                file.write_text("parallel-web-search: " + value)
                with self.assertRaises(ValueError):
                    workflow.parse_yaml(file)
            with self.assertRaises(ValueError):
                workflow.LaboratoryWorkflow("topic", None, parallel_web_search="false")


class TransportTests(unittest.TestCase):
    def test_discovery_uses_project_user_agent_and_anonymous_endpoint(self):
        transport = RecordingMCP()
        with patch("parallel_search.urlopen", transport):
            self.assertEqual(len(ParallelSearch().list_tools()["tools"]), 2)
        for req, _, _ in transport.requests:
            self.assertEqual(req.full_url, MCP_URL)
            self.assertEqual(req.get_header("User-agent"), USER_AGENT)
            self.assertIsNone(req.get_header("Authorization"))

    def test_bad_rpc_tool_and_payload_responses_are_rejected(self):
        values = [b"not json", b"[]", b'{"jsonrpc":"2.0","id":99,"result":{}}',
                  b'{"jsonrpc":"2.0","id":2,"error":{"code":-1}}',
                  b'{"jsonrpc":"2.0","id":2,"result":{"isError":true}}',
                  b'{"jsonrpc":"2.0","id":2,"result":{"content":[]}}']
        for payload in values:
            transport = RecordingMCP()
            def bad_response(req, timeout):
                if json.loads(req.data)["method"] == "tools/call":
                    return Response(payload, raw=True)
                return transport(req, timeout)
            with patch("parallel_search.urlopen", bad_response), self.assertRaises(ParallelSearchError):
                ParallelSearch().search("query", "run-id")

    def test_output_is_bounded(self):
        client = ParallelSearch()
        with patch.object(client, "_rpc", return_value={"content": [{"type": "text", "text": "x" * 30_000}]}):
            result = client.search("query", "run-id")
            self.assertLess(len(result), 20_200)
            self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
