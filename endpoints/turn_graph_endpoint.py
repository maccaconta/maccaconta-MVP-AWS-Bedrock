# endpoints/turn_graph_endpoint.py
# =============================================================================
# Endpoint paralelo ao /turn atual:
#   POST /v1/ai/turn-graph
# Objetivo: executar o fluxo do Turn via LangGraph, sem tocar nos endpoints atuais.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse

from graphs.turn_graph import run_turn_graph

router = APIRouter()


@router.post("/turn-graph")
async def post_turn_graph(
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    if payload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Payload JSON inválido"},
        )

    # roda o grafo
    final_state = run_turn_graph(payload, vars(request.app.state.config).copy())

    if final_state.get("error"):
        return JSONResponse(
            status_code=int(final_state.get("http_status", 500)),
            content={"error": final_state["error"]},
        )

    return JSONResponse(
        status_code=200,
        content=final_state["response"],
    )