# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Shopsense Env Environment.

This module creates an HTTP server that exposes the ShopsenseEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ShopsenseAction, ShopsenseObservation
    from .shopsense_env_environment import ShopsenseEnvironment
except (ImportError, SystemError, Exception):
    from models import ShopsenseAction, ShopsenseObservation
    from server.shopsense_env_environment import ShopsenseEnvironment


# Create the app with web interface and README integration
app = create_app(
    ShopsenseEnvironment,
    ShopsenseAction,
    ShopsenseObservation,
    env_name="shopsense_env",
    max_concurrent_envs=4,
)


@app.get("/")
@app.get("/health")
def root():
    """Health check — judges auto-ping this; must return HTTP 200."""
    return {
        "name": "shopsense_env",
        "status": "ok",
        "version": "0.1.0",
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m shopsense_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn shopsense_env.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
