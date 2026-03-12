# endpoints/turn_graph_endpoint.py
# =============================================================================
# Endpoint paralelo ao /turn atual:
#   POST /v1/ai/turn-graph
# Objetivo: executar o fluxo do Turn via LangGraph, sem tocar nos endpoints atuais.
# =============================================================================

from flask import Blueprint, request, jsonify, current_app
from graphs.turn_graph import run_turn_graph

turn_graph_bp = Blueprint("turn_graph", __name__)

@turn_graph_bp.route("/turn-graph", methods=["POST"])
def post_turn_graph():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON inválido"}), 400

    # roda o grafo
    final_state = run_turn_graph(payload, dict(current_app.config))

    if final_state.get("error"):
        return jsonify({"error": final_state["error"]}), int(final_state.get("http_status", 500))

    return jsonify(final_state["response"]), 200