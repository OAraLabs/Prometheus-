# Provenance: Inspired by anthropics/skills frontend-design + web-artifacts-builder
# License: MIT

"""Serve an HTML dashboard on a local HTTP server."""

from __future__ import annotations

import asyncio
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import tempfile

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class DashboardInput(BaseModel):
    """Arguments for serving an HTML dashboard."""

    html: str = Field(description="HTML content to serve")
    port: int = Field(default=8080, ge=1024, le=65535, description="Port to serve on")
    title: str | None = Field(default=None, description="Optional page title")


class DashboardTool(BaseTool):
    """Serve HTML content on a local HTTP server for visualization."""

    name = "dashboard"
    description = (
        "Start a lightweight HTTP server that serves the provided HTML content. "
        "Returns the URL to view the dashboard. Useful for visualizations, "
        "reports, and interactive HTML pages."
    )
    input_model = DashboardInput

    _servers: dict[int, HTTPServer] = {}

    async def execute(
        self, arguments: DashboardInput, context: ToolExecutionContext
    ) -> ToolResult:
        port = arguments.port
        html_content = arguments.html

        if arguments.title:
            if "<title>" not in html_content.lower():
                html_content = html_content.replace(
                    "<head>", f"<head><title>{arguments.title}</title>", 1
                )

        # Write HTML to a temp directory
        tmpdir = Path(tempfile.mkdtemp(prefix="prometheus-dash-"))
        index = tmpdir / "index.html"
        index.write_text(html_content, encoding="utf-8")

        # Stop any existing server on this port
        if port in self._servers:
            self._servers[port].shutdown()
            del self._servers[port]

        # Start HTTP server in a background thread
        handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        try:
            server = HTTPServer(("0.0.0.0", port), handler)
        except OSError as exc:
            return ToolResult(
                output=f"Cannot bind to port {port}: {exc}", is_error=True
            )

        self._servers[port] = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = f"http://localhost:{port}/"
        return ToolResult(
            output=f"Dashboard serving at {url}\nHTML file: {index}",
            metadata={"url": url, "port": port, "html_path": str(index)},
        )
