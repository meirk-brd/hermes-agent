"""Tests for Bright Data web backend integration.

Coverage:
  _get_brightdata_api_key() — API key handling, missing key error.
  _get_brightdata_zone() — zone resolution from env, config, default.
  _brightdata_request() — request construction, retry logic, error headers.
  _brightdata_search() — search response parsing, URL encoding, edge cases.
  _brightdata_extract() — extract response handling, per-URL errors.
  web_search_tool / web_extract_tool / web_crawl_tool — Bright Data dispatch paths.
  _is_backend_available / check_web_api_key — availability with BRIGHTDATA_API_KEY.
  _get_backend — config and fallback selection with brightdata.
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock


# ─── _get_brightdata_api_key ────────────────────────────────────────────────

class TestBrightdataApiKey:
    """Test suite for the _get_brightdata_api_key helper."""

    def test_returns_key_when_set(self):
        """BRIGHTDATA_API_KEY set → returns trimmed key."""
        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "  test-key  "}):
            from tools.web_tools import _get_brightdata_api_key
            assert _get_brightdata_api_key() == "test-key"

    def test_raises_without_api_key(self):
        """No BRIGHTDATA_API_KEY → ValueError with guidance."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRIGHTDATA_API_KEY", None)
            from tools.web_tools import _get_brightdata_api_key
            with pytest.raises(ValueError, match="BRIGHTDATA_API_KEY"):
                _get_brightdata_api_key()

    def test_empty_string_raises(self):
        """BRIGHTDATA_API_KEY='' → ValueError."""
        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": ""}):
            from tools.web_tools import _get_brightdata_api_key
            with pytest.raises(ValueError, match="BRIGHTDATA_API_KEY"):
                _get_brightdata_api_key()


# ─── _get_brightdata_zone ───────────────────────────────────────────────────

class TestBrightdataZone:
    """Test suite for zone resolution."""

    def test_env_var_takes_priority(self):
        """BRIGHTDATA_UNLOCKER_ZONE env → used over config and default."""
        with patch.dict(os.environ, {"BRIGHTDATA_UNLOCKER_ZONE": "my_zone"}):
            with patch("tools.web_tools._load_web_config", return_value={"brightdata_zone": "config_zone"}):
                from tools.web_tools import _get_brightdata_zone
                assert _get_brightdata_zone() == "my_zone"

    def test_config_fallback(self):
        """No env var, web.brightdata_zone in config → used."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRIGHTDATA_UNLOCKER_ZONE", None)
            with patch("tools.web_tools._load_web_config", return_value={"brightdata_zone": "config_zone"}):
                from tools.web_tools import _get_brightdata_zone
                assert _get_brightdata_zone() == "config_zone"

    def test_default_zone(self):
        """No env var, no config → default 'cli_unlocker'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRIGHTDATA_UNLOCKER_ZONE", None)
            with patch("tools.web_tools._load_web_config", return_value={}):
                from tools.web_tools import _get_brightdata_zone
                assert _get_brightdata_zone() == "cli_unlocker"


# ─── _brightdata_request ────────────────────────────────────────────────────

class TestBrightdataRequest:
    """Test suite for the _brightdata_request helper."""

    def test_posts_with_bearer_auth(self):
        """Request uses Bearer token and correct headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}):
            with patch("tools.web_tools.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_tools import _brightdata_request
                _brightdata_request({"zone": "cli_unlocker", "url": "https://example.com"})

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args
                headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
                assert headers["Authorization"] == "Bearer test-key"
                assert headers["User-Agent"] == "hermes-agent"
                assert "api.brightdata.com/request" in call_kwargs.args[0]

    def test_returns_text_when_not_json(self):
        """Non-JSON content-type → returns response.text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "markdown content here"
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}):
            with patch("tools.web_tools.httpx.post", return_value=mock_response):
                from tools.web_tools import _brightdata_request
                result = _brightdata_request({"zone": "z", "url": "https://example.com"})
                assert result == "markdown content here"

    def test_raises_on_brd_error_header(self):
        """x-brd-error header → ValueError with the error message."""
        import httpx as _httpx
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"x-brd-error": "zone not found"}
        mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "400", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}):
            with patch("tools.web_tools.httpx.post", return_value=mock_response):
                from tools.web_tools import _brightdata_request
                with pytest.raises(ValueError, match="zone not found"):
                    _brightdata_request({"zone": "z", "url": "https://example.com"})


# ─── _brightdata_search ─────────────────────────────────────────────────────

class TestBrightdataSearch:
    """Test search response parsing."""

    def _mock_request(self, return_value):
        """Helper to patch _brightdata_request and interrupt check."""
        return patch("tools.web_tools._brightdata_request", return_value=return_value)

    def test_parses_organic_results(self):
        """Standard Google SERP JSON → normalized web results."""
        raw = {
            "organic": [
                {"link": "https://example.com", "title": "Example", "description": "A site", "rank": 1},
                {"link": "https://other.com", "title": "Other", "description": "Another", "rank": 2},
            ]
        }
        with self._mock_request(raw), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test query", limit=5)
            assert result["success"] is True
            web = result["data"]["web"]
            assert len(web) == 2
            assert web[0]["title"] == "Example"
            assert web[0]["url"] == "https://example.com"
            assert web[0]["description"] == "A site"
            assert web[0]["position"] == 1

    def test_parses_text_json_response(self):
        """API returns text/plain with JSON content → parsed correctly."""
        raw_text = json.dumps({
            "organic": [
                {"link": "https://r.com", "title": "Result", "description": "Desc"},
            ]
        })
        with self._mock_request(raw_text), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test", limit=5)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"

    def test_skips_entries_without_link_or_title(self):
        """Entries missing link or title are filtered out."""
        raw = {
            "organic": [
                {"link": "", "title": "No Link", "description": "x"},
                {"link": "https://ok.com", "title": "", "description": "x"},
                {"link": "https://ok.com", "title": "Good", "description": "x"},
            ]
        }
        with self._mock_request(raw), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test", limit=5)
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Good"

    def test_empty_organic(self):
        """No organic results → empty web list."""
        with self._mock_request({"organic": []}), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test", limit=5)
            assert result["success"] is True
            assert result["data"]["web"] == []

    def test_unparseable_text_returns_empty(self):
        """Non-JSON text response → empty results (no crash)."""
        with self._mock_request("This is not JSON"), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test", limit=5)
            assert result["success"] is True
            assert result["data"]["web"] == []

    def test_respects_limit(self):
        """Results are capped at the requested limit."""
        raw = {"organic": [
            {"link": f"https://r{i}.com", "title": f"R{i}", "description": "x"}
            for i in range(10)
        ]}
        with self._mock_request(raw), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_search
            result = _brightdata_search("test", limit=3)
            assert len(result["data"]["web"]) == 3

    def test_request_includes_parsed_light(self):
        """Search request must include data_format=parsed_light."""
        with patch("tools.web_tools._get_brightdata_api_key", return_value="k"), \
             patch("tools.web_tools._get_brightdata_zone", return_value="z"), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"organic": []}
            mock_response.raise_for_status = MagicMock()

            with patch("tools.web_tools.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_tools import _brightdata_search
                _brightdata_search("hello world", limit=5)
                payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
                assert payload["data_format"] == "parsed_light"
                assert payload["zone"] == "z"
                assert "brd_json=1" in payload["url"]
                assert "hello+world" in payload["url"] or "hello%20world" in payload["url"]


# ─── _brightdata_extract ────────────────────────────────────────────────────

class TestBrightdataExtract:
    """Test extract response handling."""

    def test_basic_extract(self):
        """Single URL → markdown content returned."""
        with patch("tools.web_tools._brightdata_request", return_value="# Hello World"), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_extract
            results = _brightdata_extract(["https://example.com"])
            assert len(results) == 1
            assert results[0]["url"] == "https://example.com"
            assert results[0]["content"] == "# Hello World"
            assert results[0]["metadata"]["sourceURL"] == "https://example.com"

    def test_extract_error_per_url(self):
        """Failed URL → error entry, other URLs still succeed."""
        call_count = 0

        def mock_request(payload, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("connection refused")
            return "# Content"

        with patch("tools.web_tools._brightdata_request", side_effect=mock_request), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_extract
            results = _brightdata_extract(["https://fail.com", "https://ok.com"])
            assert len(results) == 2
            assert "error" in results[0]
            assert "connection refused" in results[0]["error"]
            assert results[1]["content"] == "# Content"

    def test_extract_json_response(self):
        """API returns dict instead of text → serialized to JSON string."""
        with patch("tools.web_tools._brightdata_request", return_value={"html": "<p>hi</p>"}), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _brightdata_extract
            results = _brightdata_extract(["https://example.com"])
            assert '"html"' in results[0]["content"]


# ─── web_search_tool (Bright Data dispatch) ─────────────────────────────────

class TestWebSearchBrightdata:
    """Test web_search_tool dispatch to Bright Data."""

    def test_search_dispatches_to_brightdata(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "organic": [{"link": "https://r.com", "title": "Result", "description": "desc"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="brightdata"), \
             patch("tools.web_tools._get_brightdata_zone", return_value="cli_unlocker"), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}), \
             patch("tools.web_tools.httpx.post", return_value=mock_response), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("test query", limit=3))
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1
            assert result["data"]["web"][0]["title"] == "Result"


# ─── web_extract_tool (Bright Data dispatch) ────────────────────────────────

class TestWebExtractBrightdata:
    """Test web_extract_tool dispatch to Bright Data."""

    def test_extract_dispatches_to_brightdata(self):
        with patch("tools.web_tools._get_backend", return_value="brightdata"), \
             patch("tools.web_tools._brightdata_extract", return_value=[{
                 "url": "https://example.com", "title": "Page",
                 "content": "# Content", "raw_content": "# Content",
                 "metadata": {"sourceURL": "https://example.com"},
             }]), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}):
            from tools.web_tools import web_extract_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["url"] == "https://example.com"


# ─── web_crawl_tool (Bright Data dispatch) ──────────────────────────────────

class TestWebCrawlBrightdata:
    """Test web_crawl_tool dispatch to Bright Data."""

    def test_crawl_dispatches_to_brightdata(self):
        with patch("tools.web_tools._get_backend", return_value="brightdata"), \
             patch("tools.web_tools._brightdata_extract", return_value=[{
                 "url": "https://example.com", "title": "",
                 "content": "# Page content", "raw_content": "# Page content",
                 "metadata": {"sourceURL": "https://example.com"},
             }]), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test-key"}):
            from tools.web_tools import web_crawl_tool
            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_crawl_tool("https://example.com", use_llm_processing=False)
            ))
            assert "results" in result
            assert len(result["results"]) == 1
            assert "Page content" in result["results"][0]["content"]


# ─── Backend selection & availability ────────────────────────────────────────

class TestBackendSelectionBrightdata:
    """Test _get_backend and availability checks with Bright Data."""

    _ENV_KEYS = (
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
        "BRIGHTDATA_API_KEY",
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TAVILY_API_KEY",
    )

    def setup_method(self):
        os.environ["HERMES_ENABLE_NOUS_MANAGED_TOOLS"] = "1"
        for key in self._ENV_KEYS:
            if key != "HERMES_ENABLE_NOUS_MANAGED_TOOLS":
                os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_config_brightdata(self):
        """web.backend=brightdata in config → 'brightdata' regardless of keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brightdata"}):
            assert _get_backend() == "brightdata"

    def test_config_brightdata_case_insensitive(self):
        """web.backend=BrightData (mixed case) → 'brightdata'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "BrightData"}):
            assert _get_backend() == "brightdata"

    def test_config_brightdata_overrides_env_keys(self):
        """web.backend=brightdata → 'brightdata' even if Firecrawl key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brightdata"}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "brightdata"

    def test_fallback_brightdata_only_key(self):
        """Only BRIGHTDATA_API_KEY set → 'brightdata'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test"}):
            assert _get_backend() == "brightdata"

    def test_fallback_firecrawl_takes_priority_over_brightdata(self):
        """Firecrawl + Bright Data keys, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_brightdata_takes_priority_over_parallel(self):
        """Bright Data + Parallel keys, no config → 'brightdata'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "brightdata"

    def test_is_backend_available_brightdata(self):
        """_is_backend_available('brightdata') checks BRIGHTDATA_API_KEY."""
        from tools.web_tools import _is_backend_available
        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test"}):
            assert _is_backend_available("brightdata") is True

    def test_is_backend_available_brightdata_missing(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("brightdata") is False

    def test_check_web_api_key_brightdata_only(self):
        """check_web_api_key() returns True with only BRIGHTDATA_API_KEY."""
        with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_check_web_api_key_configured_brightdata(self):
        """web.backend=brightdata + key set → True."""
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brightdata"}), \
             patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "bd-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_check_web_api_key_configured_brightdata_no_key(self):
        """web.backend=brightdata but no key → False."""
        with patch("tools.web_tools._load_web_config", return_value={"backend": "brightdata"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is False


def test_web_requires_env_includes_brightdata_key():
    from tools.web_tools import _web_requires_env
    assert "BRIGHTDATA_API_KEY" in _web_requires_env()
