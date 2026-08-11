"""
API ENTRYPOINT
================
Creates the FastAPI app and wires in the routers from routes.py and
streaming.py. Kept minimal on purpose -- this file's only job is app
setup (CORS, router registration), not endpoint logic.

Run with:
    uvicorn api.main:app --reload

Then:
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "Did the Feds language about inflation change between January and June 2026?"}'

    curl -N -X POST http://localhost:8000/stream/query \
      -H "Content-Type: application/json" \
      -d '{"question": "Did the January minutes reveal disagreement?"}'
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as query_router
from api.streaming import router as streaming_router

app = FastAPI(title="FOMC Agentic RAG API")

# CORS needed so the frontend/ static files (served from a different
# origin/port during local development) can call this API from the
# browser without being blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(streaming_router)
