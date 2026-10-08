"""Serve the studio UI on a separate hostname while sharing the live backend."""

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app_live_web import app as live_app
from lam.live.settings import settings


studio_host = os.getenv("LAM_STUDIO_HOST", "studio.localhost").lower().rstrip(".")
studio_app = FastAPI()
studio_app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "webgl_frontend_alt" / "dist", html=True),
    name="studio",
)


async def app(scope, receive, send):
    host = next(
        (value.decode("latin-1").split(":", 1)[0].lower().rstrip(".")
         for key, value in scope.get("headers", []) if key == b"host"),
        "",
    )
    path = scope.get("path", "/")
    is_backend_route = any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in ("/api", "/ws", settings.oac_assets_route.rstrip("/"))
    )
    target = studio_app if host == studio_host and scope["type"] == "http" and not is_backend_route else live_app
    await target(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run("app_live_studio:app", host="127.0.0.1", port=7861, ws_ping_interval=30, ws_ping_timeout=120)
