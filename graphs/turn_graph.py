# graphs/turn_graph.py
# =============================================================================
# LangGraph Turn Flow (versão paralela ao endpoint /turn atual)
#
# Objetivo:
# - Reusar serviços e templates existentes:
#   PromptRepository, compose_turn_prompt, BedrockKnowledgeBaseService, BedrockRuntimeService
# - Organizar o fluxo em nós (validate -> load templates -> retrieve -> compose -> invoke -> response)
# - NÃO altera nenhum script atual: é uma implementação paralela para /turn-graph
#
# Requisito:
#   pip install langgraph langchain-core
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

import time


# -----------------------------------------------------------------------------
# State do Grafo
# -----------------------------------------------------------------------------
class TurnState(TypedDict, total=False):
    # inputs
    payload: Dict[str, Any]

    # templates
    blueprint: Dict[str, Any]
    components: Dict[str, Any]

    # rag
    kb_id: Optional[str]
    kb_raw: Dict[str, Any]
    rag: Dict[str, Any]

    # prompt + model
    prompt: str
    model_id: str
    generation: Dict[str, Any]
    llm: Dict[str, Any]

    # output final
    response: Dict[str, Any]

    # erro
    error: str
    http_status: int


# -----------------------------------------------------------------------------
# Imports do seu projeto (ajuste somente se seus caminhos forem diferentes)
# -----------------------------------------------------------------------------
from utils.validation_utils import validate_turn_payload
from utils.prompt_repository import PromptRepository
from utils.prompt_composer import compose_turn_prompt

from services.bedrock_kb_service import BedrockKnowledgeBaseService
from services.bedrock_runtime_service import BedrockRuntimeService


# -----------------------------------------------------------------------------
# Inicialização / cache simples (MVP)
# -----------------------------------------------------------------------------
_repo: Optional[PromptRepository] = None
_kb: Optional[BedrockKnowledgeBaseService] = None
_rt: Optional[BedrockRuntimeService] = None


def _app_cfg_from_runnable_config(config: RunnableConfig) -> Dict[str, Any]:
    """
    Recupera o current_app.config (que a gente passa pelo endpoint) via RunnableConfig.
    """
    cfg = config.get("configurable", {}) if config else {}
    app_cfg = cfg.get("app_config", {})
    if not isinstance(app_cfg, dict):
        return {}
    return app_cfg


def init_services(app_config: Dict[str, Any]) -> None:
    """
    Inicializa dependências compartilhadas (reuso em todas as chamadas).
    """
    global _repo, _kb, _rt

    if _repo is None:
        _repo = PromptRepository(app_config["TEMPLATES_ROOT"])

    if _kb is None:
        _kb = BedrockKnowledgeBaseService(
            region=app_config.get("AWS_REGION", "us-east-1"),
            timeout_seconds=int(app_config.get("BEDROCK_TIMEOUT_SECONDS", 20)),
        )

    if _rt is None:
        _rt = BedrockRuntimeService(
            region=app_config.get("AWS_REGION", "us-east-1"),
            timeout_seconds=int(app_config.get("BEDROCK_TIMEOUT_SECONDS", 20)),
        )


# -----------------------------------------------------------------------------
# Nós do grafo
# -----------------------------------------------------------------------------
def node_validate(state: TurnState, config: RunnableConfig) -> TurnState:
    payload = state.get("payload")
    if not isinstance(payload, dict):
        return {"error": "Payload JSON inválido", "http_status": 400}

    err = validate_turn_payload(payload)
    if err:
        return {"error": err, "http_status": 400}

    return {}


def node_load_templates(state: TurnState, config: RunnableConfig) -> TurnState:
    assert _repo is not None, "PromptRepository não inicializado"
    payload = state["payload"]

    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    try:
        blueprint = _repo.load_blueprint(blueprint_id, blueprint_version)
    except FileNotFoundError as e:
        return {"error": str(e), "http_status": 404}

    refs = blueprint.get("componentRefs", {})
    try:
        components = {
            "persona": _repo.load_component("personas", refs["persona"]),
            "especialidade": _repo.load_component("especialidades", refs["especialidade"]),
            "cenario": _repo.load_component("cenarios", refs["cenario"]),
            "politicas": _repo.load_component("politicas", refs["politicas"]),
            "saida": _repo.load_component("saida", refs["saida"]),
        }
    except KeyError as e:
        return {"error": f"Blueprint sem componentRefs obrigatório: {e}", "http_status": 500}
    except FileNotFoundError as e:
        return {"error": str(e), "http_status": 404}

    return {"blueprint": blueprint, "components": components}


# def node_retrieve_kb(state: TurnState, config: RunnableConfig) -> TurnState:
    assert _kb is not None, "BedrockKnowledgeBaseService não inicializado"
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    kb_id = app_cfg.get("BEDROCK_KB_ID")

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", app_cfg.get("DEFAULT_TOP_K", 3)))
    threshold = float(retrieval_cfg.get("scoreThreshold", app_cfg.get("DEFAULT_SCORE_THRESHOLD", 0.1)))
    filters = retrieval_cfg.get("filters")

    # Se não tiver KB configurada, não travamos: seguimos sem RAG (útil em DEV).
    if not kb_id:
        rag_norm = {
            "evidences": [],
            "citations": [],
            "flags": ["no_kb_configured"],
            "no_evidence": True,
        }
        return {"kb_id": None, "rag": rag_norm}

    try:
        kb_resp = _kb.retrieve(
            knowledge_base_id=kb_id,
            query_text=payload["userText"],
            top_k=top_k,
            filters=filters,
        )
        rag_norm = _kb.normalize(kb_resp, score_threshold=threshold)
    except Exception as e:
        return {"error": f"Falha no retrieve da KB: {e}", "http_status": 502}

    return {"kb_id": kb_id, "kb_raw": kb_resp, "rag": rag_norm}

def node_retrieve_kb(state: TurnState, config: RunnableConfig) -> TurnState:
    assert _kb is not None, "BedrockKnowledgeBaseService não inicializado"
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    kb_id = app_cfg.get("BEDROCK_KB_ID")

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", app_cfg.get("DEFAULT_TOP_K", 3)))
    threshold = float(retrieval_cfg.get("scoreThreshold", app_cfg.get("DEFAULT_SCORE_THRESHOLD", 0.1)))
    filters = retrieval_cfg.get("filters")

    t0 = time.perf_counter()

    # Sem KB: segue sem RAG, mas mantém contrato
    if not kb_id:
        rag_norm = {
            "evidences": [],
            "citations": [],
            "flags": [],
            "no_evidence": True,
        }
        retrieve_ms = int((time.perf_counter() - t0) * 1000)

        rag_block = {
            "avgScore": 0.0,
            "maxScore": 0.0,
            "evidenceCount": 0,
            "citations": [],
            "flags": [],
            "noEvidence": True,
        }

        return {
            "kb_id": None,
            "rag": rag_norm,          # normalizado (interno)
            "rag_block": rag_block,   # bloco no formato do endpoint antigo
            "retrieve_ms": retrieve_ms,
        }

    try:
        kb_resp = _kb.retrieve(
            knowledge_base_id=kb_id,
            query_text=payload["userText"],
            top_k=top_k,
            filters=filters,
        )
        rag_norm = _kb.normalize(kb_resp, score_threshold=threshold)
    except Exception as e:
        return {"error": f"Falha no retrieve da KB: {e}", "http_status": 502}

    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    # Calcula stats do RAG a partir de evidences normalizadas
    evidences = rag_norm.get("evidences", []) or []
    scores = [float(ev.get("score", 0.0)) for ev in evidences]
    evidence_count = len(evidences)
    max_score = max(scores) if scores else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    rag_block = {
        "avgScore": avg_score,
        "maxScore": max_score,
        "evidenceCount": evidence_count,
        "citations": rag_norm.get("citations", []),
        "flags": rag_norm.get("flags", []),
        "noEvidence": bool(rag_norm.get("no_evidence")),
    }

    return {
        "kb_id": kb_id,
        "kb_raw": kb_resp,
        "rag": rag_norm,
        "rag_block": rag_block,
        "retrieve_ms": retrieve_ms,
    }

def node_compose_prompt(state: TurnState, config: RunnableConfig) -> TurnState:
    payload = state["payload"]
    prompt_str = compose_turn_prompt(
        payload_json=payload,
        blueprint_json=state["blueprint"],
        components_json=state["components"],
        rag_json=state.get("rag", {}),
    )
    return {"prompt": prompt_str}


# def node_invoke_model(state: TurnState, config: RunnableConfig) -> TurnState:
    assert _rt is not None, "BedrockRuntimeService não inicializado"
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    gen_cfg = payload.get("generationConfig", {})
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", app_cfg.get("DEFAULT_MAX_OUTPUT_TOKENS", 600))),
        "temperature": float(gen_cfg.get("temperature", app_cfg.get("DEFAULT_TEMPERATURE", 0.2))),
        "topP": float(gen_cfg.get("topP", app_cfg.get("DEFAULT_TOP_P", 0.9))),
    }

    model_id = app_cfg.get("BEDROCK_TURN_MODEL_ID", "amazon.nova-pro-v1:0")

    try:
        llm = _rt.invoke_text_model(
            model_id=model_id,
            prompt=state["prompt"],
            generation=generation,
        )
    except Exception as e:
        return {"error": f"Falha ao invocar modelo: {e}", "http_status": 502}

    return {"model_id": model_id, "generation": generation, "llm": llm}


# def node_build_response(state: TurnState, config: RunnableConfig) -> TurnState:
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    rag = state.get("rag", {})
    llm = state.get("llm", {})

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", app_cfg.get("DEFAULT_TOP_K", 3)))
    threshold = float(retrieval_cfg.get("scoreThreshold", app_cfg.get("DEFAULT_SCORE_THRESHOLD", 0.1)))

    resp = {
        "replyText": llm.get("replyText", ""),
        "citations": rag.get("citations", []),
        "flags": rag.get("flags", []),
        "model": llm.get("modelId", state.get("model_id")),
        "telemetry": {
            "blueprintId": blueprint_id,
            "blueprintVersion": blueprint_version,
            "usedKb": bool(state.get("kb_id")),
            "noEvidence": bool(rag.get("no_evidence")),
            "ragTopK": top_k,
            "ragThreshold": threshold,
            # Se o BedrockRuntimeService retorna usage/tokens, persistimos aqui:
            "usage": llm.get("usage"),
        },
    }
    return {"response": resp}

# def node_build_response(state: TurnState, config: RunnableConfig) -> TurnState:
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]
    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    rag = state.get("rag", {})
    llm = state.get("llm", {})

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", app_cfg.get("DEFAULT_TOP_K", 3)))
    threshold = float(retrieval_cfg.get("scoreThreshold", app_cfg.get("DEFAULT_SCORE_THRESHOLD", 0.1)))

    # telemetry básico (adicione tokens/usage se seu runtime retornar)
    telemetry = {
        "blueprintId": blueprint_id,
        "blueprintVersion": blueprint_version,
        "usedKb": bool(state.get("kb_id")),
        "noEvidence": bool(rag.get("no_evidence")),
        "ragTopK": top_k,
        "ragThreshold": threshold,
        # inclusão de metadados de sessão/turn para persistência
        "turnId": payload.get("turnId"),
        "turnIndex": payload.get("turnIndex"),
        "sessionId": payload.get("sessionId"),
    }

    # incluir usage se existir (tokens/cost)
    if isinstance(llm.get("usage"), dict):
        telemetry["usage"] = llm.get("usage")

    resp = {
        "replyText": llm.get("replyText", ""),
        "citations": rag.get("citations", []),
        "flags": rag.get("flags", []),
        "model": llm.get("modelId", state.get("model_id")),
        "telemetry": telemetry
    }
    return {"response": resp}

def node_invoke_model(state: TurnState, config: RunnableConfig) -> TurnState:
    assert _rt is not None, "BedrockRuntimeService não inicializado"
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    gen_cfg = payload.get("generationConfig", {})
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", app_cfg.get("DEFAULT_MAX_OUTPUT_TOKENS", 700))),
        "temperature": float(gen_cfg.get("temperature", app_cfg.get("DEFAULT_TEMPERATURE", 0.2))),
        "topP": float(gen_cfg.get("topP", app_cfg.get("DEFAULT_TOP_P", 0.9))),
    }

    model_id = app_cfg.get("BEDROCK_TURN_MODEL_ID", "amazon.nova-pro-v1:0")

    t0 = time.perf_counter()
    try:
        llm = _rt.invoke_text_model(
            model_id=model_id,
            prompt=state["prompt"],
            generation=generation,
        )
    except Exception as e:
        return {"error": f"Falha ao invocar modelo: {e}", "http_status": 502}

    invoke_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "model_id": model_id,
        "generation": generation,
        "llm": llm,
        "invoke_ms": invoke_ms,
    }

def node_build_response(state: TurnState, config: RunnableConfig) -> TurnState:
    app_cfg = _app_cfg_from_runnable_config(config)
    payload = state["payload"]

    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    rag_norm = state.get("rag", {}) or {}
    rag_block = state.get("rag_block", {}) or {}

    llm = state.get("llm", {}) or {}
    prompt_str = state.get("prompt", "") or ""

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", app_cfg.get("DEFAULT_TOP_K", 3)))
    threshold = float(retrieval_cfg.get("scoreThreshold", app_cfg.get("DEFAULT_SCORE_THRESHOLD", 0.1)))

    # === tokens ===
    # Tentamos obter do rawModelResponse.usage como no seu endpoint antigo.
    raw_model = llm.get("rawModelResponse") or llm.get("raw") or llm.get("raw_response")
    usage = None
    if isinstance(raw_model, dict):
        usage = raw_model.get("usage")

    # fallback: se seu runtime já devolve usage separado
    if usage is None and isinstance(llm.get("usage"), dict):
        usage = llm["usage"]

    tokens_in = None
    tokens_out = None
    total_tokens = None
    if isinstance(usage, dict):
        tokens_in = usage.get("inputTokens")
        tokens_out = usage.get("outputTokens")
        total_tokens = usage.get("totalTokens")

    # === execution ===
    retrieve_ms = int(state.get("retrieve_ms", 0) or 0)
    invoke_ms = int(state.get("invoke_ms", 0) or 0)
    latency_ms = retrieve_ms + invoke_ms

    # custo: só calcule se você já tem uma tabela interna de preços.
    # Para manter o contrato, vamos preencher apenas se houver.
    cost_estimate = llm.get("costEstimateUsd")

    execution = {
        "latencyMs": latency_ms,
        "tokensIn": tokens_in,
        "tokensOut": tokens_out,
        "totalTokens": total_tokens,
        "costEstimateUsd": cost_estimate,
    }

    # === promptMetadata ===
    prompt_meta = {
        "blueprintId": blueprint_id,
        "blueprintVersion": blueprint_version,
        "promptLengthChars": len(prompt_str),
        "promptLengthTokens": tokens_in,  # no seu endpoint antigo bate com inputTokens
    }

    # === resposta final (igual ao endpoint antigo) ===
    resp = {
        "citations": rag_norm.get("citations", []),
        "execution": execution,
        "flags": rag_norm.get("flags", []),
        "generationConfig": state.get("generation", {}),
        "model": llm.get("modelId", state.get("model_id")),
        "promptMetadata": prompt_meta,
        "rag": rag_block,
        "rawModelResponse": raw_model if isinstance(raw_model, dict) else llm.get("rawModelResponse"),
        "replyText": llm.get("replyText", ""),
        "telemetry": {
            "blueprintId": blueprint_id,
            "blueprintVersion": blueprint_version,
            "ragThreshold": threshold,
            "ragTopK": top_k,
            "usedKb": bool(state.get("kb_id")),
        },
    }

    # Remove chaves None para ficar limpo (opcional)
    if resp["execution"]["tokensIn"] is None:
        # se você preferir manter explícito, remova este bloco
        pass

    return {"response": resp}


# -----------------------------------------------------------------------------
# Roteamento de erro (condicional)
# -----------------------------------------------------------------------------
def route_if_error(state: TurnState) -> str:
    return "error" if state.get("error") else "ok"


def node_error(state: TurnState, config: RunnableConfig) -> TurnState:
    # Mantém error/http_status; nada a fazer aqui.
    if "http_status" not in state:
        state["http_status"] = 500
    return {}


# -----------------------------------------------------------------------------
# Builder do Grafo
# -----------------------------------------------------------------------------
def build_turn_graph():
    g = StateGraph(TurnState)

    g.add_node("validate", node_validate)
    g.add_node("load_templates", node_load_templates)
    g.add_node("retrieve_kb", node_retrieve_kb)
    g.add_node("compose_prompt", node_compose_prompt)
    g.add_node("invoke_model", node_invoke_model)
    g.add_node("build_response", node_build_response)
    g.add_node("error", node_error)

    g.set_entry_point("validate")

    g.add_conditional_edges("validate", route_if_error, {"ok": "load_templates", "error": "error"})
    g.add_conditional_edges("load_templates", route_if_error, {"ok": "retrieve_kb", "error": "error"})
    g.add_conditional_edges("retrieve_kb", route_if_error, {"ok": "compose_prompt", "error": "error"})
    g.add_conditional_edges("compose_prompt", route_if_error, {"ok": "invoke_model", "error": "error"})
    g.add_conditional_edges("invoke_model", route_if_error, {"ok": "build_response", "error": "error"})

    g.add_edge("build_response", END)
    g.add_edge("error", END)

    return g.compile()


# Singleton do grafo compilado (cache)
_turn_graph = None


def run_turn_graph(payload: Dict[str, Any], app_config: Dict[str, Any]) -> TurnState:
    """
    Executa o grafo e retorna o state final.

    Parâmetros:
      payload: JSON recebido pelo endpoint (mesmo payload do /turn).
      app_config: dict(current_app.config) do Flask.

    Retorno:
      - Sucesso: {"response": {...}, ...}
      - Falha:   {"error": "...", "http_status": N, ...}
    """
    global _turn_graph

    # init serviços com config do Flask
    init_services(app_config)

    # compila grafo 1x
    if _turn_graph is None:
        _turn_graph = build_turn_graph()

    # RunnableConfig é o formato correto esperado por LangGraph.
    runnable_config: RunnableConfig = {
        "configurable": {
            "app_config": app_config
        }
    }

    initial_state: TurnState = {"payload": payload}

    # Agora o LangGraph chamará os nós com (state, config) corretamente.
    final_state: TurnState = _turn_graph.invoke(initial_state, config=runnable_config)
    return final_state