"""ML Hub — single FastAPI app serving all ML modules + a landing UI."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .modules import cloudlens, faceid

BASE = Path(__file__).parent
app = FastAPI(title="ML Hub", description="Hub demo project ML")
app.include_router(cloudlens.router)
app.include_router(faceid.router)

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

PROJECTS = [
    {
        "key":   "cloudlens",
        "name":  "CloudLens",
        "desc":  "Klasifikasi 7 jenis awan dari foto langit",
        "icon":  "☁",
        "url":   "/cloudlens",
        "color": "#3a80ff",
    },
    {
        "key":   "faceid",
        "name":  "FaceID",
        "desc":  "Pengenalan wajah + deteksi emosi",
        "icon":  "👤",
        "url":   "/faceid",
        "color": "#ff6b6b",
    },
]


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request, "index.html", {"projects": PROJECTS})


@app.get("/cloudlens", response_class=HTMLResponse)
async def cloudlens_page(request: Request):
    return templates.TemplateResponse(request, "cloudlens.html")


@app.get("/faceid", response_class=HTMLResponse)
async def faceid_page(request: Request):
    return templates.TemplateResponse(request, "faceid.html")
