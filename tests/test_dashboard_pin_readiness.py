"""Provider pin must wait for the loopback dashboard, not race its cold start.

``Dashboard.start()`` spawns ``hermes dashboard`` and returns; the process needs
seconds before it binds its port. The pin in ``api_config_put`` fires
immediately afterwards, so on a cold or restarted dashboard it used to fail with
"Could not reach the Hermes dashboard to pin <provider> (All connection attempts
failed)" on a save that had actually succeeded, leaving hermes on
auto-resolution.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx


ROOT = Path(__file__).resolve().parents[1]


class FakeRequest:
    """Just enough of a Starlette Request for api_config_put."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


class PinReadinessFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hermes-pin-readiness-")
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name) / ".hermes"
        home.mkdir()
        env = patch.dict(os.environ, {
            "HERMES_HOME": str(home),
            "ADMIN_PASSWORD": "test-password",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        spec = importlib.util.spec_from_file_location(
            f"hermes_pin_readiness_{uuid.uuid4().hex}", ROOT / "server.py"
        )
        assert spec and spec.loader
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        self.server = server

    @staticmethod
    def _response(status_code: int) -> SimpleNamespace:
        return SimpleNamespace(status_code=status_code)

    @staticmethod
    def _client(side_effect) -> SimpleNamespace:
        return SimpleNamespace(get=AsyncMock(side_effect=side_effect))


class DashboardReadinessTests(PinReadinessFixture):
    async def test_answering_dashboard_is_ready(self):
        client = self._client([self._response(200)])

        with patch.object(self.server, "get_http_client", return_value=client):
            self.assertTrue(await self.server.wait_for_dashboard_ready(1.0))

        self.assertEqual(
            client.get.await_args.args[0],
            f"{self.server.HERMES_DASHBOARD_URL}/api/status",
        )
        self.assertEqual(client.get.await_count, 1)

    async def test_auth_gated_401_still_counts_as_ready(self):
        """A 401 proves the port is bound; the gate is hermes' own, not ours."""
        client = self._client([self._response(401)])

        with patch.object(self.server, "get_http_client", return_value=client):
            self.assertTrue(await self.server.wait_for_dashboard_ready(1.0))

    async def test_closed_port_keeps_polling_then_reports_not_ready(self):
        client = self._client(httpx.ConnectError("All connection attempts failed"))

        with patch.object(self.server, "get_http_client", return_value=client):
            started = time.monotonic()
            ready = await self.server.wait_for_dashboard_ready(0.2)
            elapsed = time.monotonic() - started

        self.assertFalse(ready)
        self.assertGreaterEqual(client.get.await_count, 2)
        self.assertLess(elapsed, 3.0)

    async def test_zero_budget_makes_exactly_one_attempt(self):
        client = self._client(httpx.ConnectError("All connection attempts failed"))

        with patch.object(self.server, "get_http_client", return_value=client):
            self.assertFalse(await self.server.wait_for_dashboard_ready(0))

        self.assertEqual(client.get.await_count, 1)

    async def test_slow_start_is_waited_out(self):
        """A dashboard that answers on the third poll is still pinned."""
        client = self._client([
            httpx.ConnectError("refused"),
            self._response(503),
            self._response(200),
        ])

        with patch.object(self.server, "get_http_client", return_value=client):
            self.assertTrue(await self.server.wait_for_dashboard_ready(5.0))

        self.assertEqual(client.get.await_count, 3)


class ConfigSavePinTests(PinReadinessFixture):
    def _save_body(self) -> dict:
        return {
            "vars": {"OPENROUTER_API_KEY": "sk-live-test-key", "LLM_MODEL": "openai/gpt-4o-mini"},
            "_active_provider_key": "OPENROUTER_API_KEY",
        }

    async def _save(self, *, ready: bool, pin_return=None):
        server = self.server
        # Auth and the real config writer are other tests' business; this one is
        # about the order of spawn → readiness → pin.
        server.guard = lambda request: None
        server.write_config_yaml = lambda *args, **kwargs: None
        server.dash = SimpleNamespace(proc=None, start=AsyncMock())
        server.wait_for_dashboard_ready = AsyncMock(return_value=ready)
        pin = AsyncMock(return_value=pin_return)
        server.set_active_model_via_hermes = pin

        response = await server.api_config_put(FakeRequest(self._save_body()))
        return json.loads(response.body), pin

    async def test_pin_runs_once_dashboard_is_ready(self):
        payload, pin = await self._save(ready=True)

        self.assertTrue(payload["ok"])
        self.assertNotIn("warning", payload)
        pin.assert_awaited_once_with(
            "openrouter", "openai/gpt-4o-mini", base_url="", api_key=""
        )

    async def test_cold_start_waits_the_cold_budget(self):
        server = self.server
        await self._save(ready=True)
        server.wait_for_dashboard_ready.assert_awaited_once_with(
            server.DASHBOARD_READY_TIMEOUT
        )
        self.server.dash.start.assert_awaited_once()

    async def test_unready_dashboard_skips_pin_and_says_why(self):
        payload, pin = await self._save(ready=False)

        pin.assert_not_awaited()
        self.assertTrue(payload["ok"], "the save itself still succeeds")
        self.assertIn("openrouter", payload["warning"])
        self.assertIn("auto-resolution", payload["warning"])

    async def test_pin_failure_still_reports_the_pin_reason(self):
        payload, _ = await self._save(ready=True, pin_return="Hermes rejected the pin (HTTP 500)")

        self.assertEqual(payload["warning"], "Hermes rejected the pin (HTTP 500)")


if __name__ == "__main__":
    unittest.main()
