# =============================================================================
# Arquivo: endpoints/turn_endpoint.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Endpoint:
#   POST /v1/ai/turn   (url_prefix "/v1/ai" é registrado no app.py)
#
# Finalidade:
#   Processar um Turn síncrono:
#     1) Valida payload
#     2) Carrega blueprint e componentes (templates)
#     3) (Opcional) Executa RAG via Bedrock KB
#     4) Compõe prompt final
#     5) Invoca modelo via Bedrock Runtime
#     6) Retorna replyText + citations + flags + telemetria
#
# Funções:
#   - Evita KeyError em config: usa current_app.config.get(...)
#   - RAG opcional: se BEDROCK_KB_ID não estiver setado, não derruba o endpoint
#   - Região e timeouts via config
#   - Defaults para top_k/threshold e geração com fallbacks
# Retorno Adicionais KPIs
# Extrai e retorna: tokensIn, tokensOut, totalTokens .
# Mede e retorna latencyMs (tempo total da invocação do modelo).
# Retorna rawModelResponse para auditoria.
# Retorna promptMetadata (blueprintId/version, promptLengthChars e tentativas de promptLengthTokens).
# Retorna rag com citations, evidenceCount, avgScore, maxScore, flags e noEvidence.
# Retorna execution.costEstimateUsd usando COST_PER_1000_TOKENS_USD da config .
# Mantém compatibilidade: continua retornando replyText, model, citations, flags, telemetry.
# Retorna telemetria completa (tokens, latência, raw, rag metrics, prompt metadata)
#
# NOTA: VER DETALHES TÉCNICOS NA DOCUMENTAÇÃO DA APLICAÇÃO
# =============================================================================
from __future__ import annotations

import time
from typing import Dict, Any, Optional

from flask import Blueprint, request, jsonify, current_app

from utils.validation_utils import validate_turn_payload
from utils.prompt_repository import PromptRepository
from utils.prompt_composer import compose_turn_prompt

from services.bedrock_kb_service import BedrockKnowledgeBaseService
from services.bedrock_runtime_service import BedrockRuntimeService

turn_bp = Blueprint("turn", __name__)

_repo: Optional[PromptRepository] = None
_kb: Optional[BedrockKnowledgeBaseService] = None
_rt: Optional[BedrockRuntimeService] = None


def _init():
    """
    Inicializa dependências compartilhadas (MVP).
    """
    global _repo, _kb, _rt

    if _repo is None:
        _repo = PromptRepository(current_app.config["TEMPLATES_ROOT"])

    if _kb is None:
        # KB é opcional: se BEDROCK_KB_ID não estiver configurado, _kb permanece None
        if current_app.config.get("BEDROCK_KB_ID"):
            _kb = BedrockKnowledgeBaseService(
                region=current_app.config["AWS_REGION"],
                timeout_seconds=current_app.config["BEDROCK_TIMEOUT_SECONDS"],
            )

    if _rt is None:
        _rt = BedrockRuntimeService(
            region=current_app.config["AWS_REGION"],
            timeout_seconds=current_app.config["BEDROCK_TIMEOUT_SECONDS"],
        )


def _extract_usage_from_raw(raw: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """
    Extrai input/output/total tokens e eventualmente custo estimado do raw retornado pelo runtime.
    Estrutura típica esperada (varia por modelo/runtime):
      raw["usage"] -> {"inputTokens": N, "outputTokens": M, "totalTokens": K}
    Faz fallback em caminhos alternativos.
    """
    usage = {"tokensIn": None, "tokensOut": None, "totalTokens": None}
    if not raw or not isinstance(raw, dict):
        return usage

    u = raw.get("usage", {}) or {}
    tokens_in = u.get("inputTokens") or u.get("input_token_count") or u.get("input_tokens")
    tokens_out = u.get("outputTokens") or u.get("output_token_count") or u.get("output_tokens")
    total = u.get("totalTokens") or u.get("total_token_count") or u.get("total_tokens")

    # ensure ints or None
    def _to_int(v):
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    usage["tokensIn"] = _to_int(tokens_in)
    usage["tokensOut"] = _to_int(tokens_out)
    usage["totalTokens"] = _to_int(total)
    return usage


def _compute_rag_metrics(rag_norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recebe o retorno normalizado do KB (normalize) e devolve métricas úteis:
      - citations (dedupe preservando ordem)
      - evidenceCount
      - avgScore, maxScore (se score presente nos evidences)
    """
    evidences = rag_norm.get("evidences", []) or []
    citations = rag_norm.get("citations", []) or []

    seen = set()
    deduped = []
    for c in citations:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    scores = []
    for ev in evidences:
        try:
            s = float(ev.get("score", 0.0))
            scores.append(s)
        except Exception:
            continue

    avg_score = sum(scores) / len(scores) if scores else None
    max_score = max(scores) if scores else None

    return {
        "citations": deduped,
        "evidenceCount": len(evidences),
        "avgScore": avg_score,
        "maxScore": max_score,
        "flags": rag_norm.get("flags", []),
        "noEvidence": rag_norm.get("no_evidence", False),
    }


@turn_bp.route("/turn", methods=["POST"])
def post_turn():
    """
    POST /v1/ai/turn
    Fluxo:
      1) valida payload
      2) carrega blueprint e componentes (templates)
      3) faz RAG (se KB configurada)
      4) compõe prompt final
      5) invoca Bedrock Runtime (medindo latência)
      6) extrai tokens/telemetria e retorna resposta completa ao backend da aplicação
    """
    _init()

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON inválido"}), 400

    err = validate_turn_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    # 1) Carrega blueprint (receita)
    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    try:
        blueprint = _repo.load_blueprint(blueprint_id, blueprint_version)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # 2) Resolve componentes do blueprint
    refs = blueprint.get("componentRefs", {})  # ex.: {"persona": "...json", ...}
    try:
        components = {
            "persona": _repo.load_component("personas", refs["persona"]),
            "especialidade": _repo.load_component("especialidades", refs["especialidade"]),
            "cenario": _repo.load_component("cenarios", refs["cenario"]),
            "politicas": _repo.load_component("politicas", refs["politicas"]),
            "saida": _repo.load_component("saida", refs["saida"]),
        }
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # 3) RAG via Knowledge Base (S3 Vectors) - opcional
    kb_id = current_app.config.get("BEDROCK_KB_ID")
    rag_norm = {"evidences": [], "citations": [], "flags": [], "no_evidence": True}
    rag_metrics = {}
    retrieval_cfg = payload.get("retrievalConfig", {}) or {}
    if kb_id and _kb:
        top_k = int(retrieval_cfg.get("topK", current_app.config.get("DEFAULT_TOP_K", 3)))
        threshold = float(retrieval_cfg.get("scoreThreshold", current_app.config.get("DEFAULT_SCORE_THRESHOLD", 0.1)))
        filters = retrieval_cfg.get("filters")
        kb_resp = _kb.retrieve(
            knowledge_base_id=kb_id,
            query_text=payload["userText"],
            top_k=top_k,
            filters=filters,
        )
        rag_norm = _kb.normalize(kb_resp, score_threshold=threshold)
        rag_metrics = _compute_rag_metrics(rag_norm)
    else:
        # se não há KB, mantenha defaults
        rag_metrics = {"citations": [], "evidenceCount": 0, "avgScore": None, "maxScore": None, "flags": [], "noEvidence": True}

    # 4) Composição do prompt final (usando seu composer existente)
    prompt_str = compose_turn_prompt(
        payload_json=payload,
        blueprint_json=blueprint,
        components_json=components,
        rag_json=rag_norm,
    )

    # Prompt metadata (chars + tokens if runtime provides prompt tokens later)
    prompt_metadata = {
        "blueprintId": blueprint_id,
        "blueprintVersion": blueprint_version,
        "promptLengthChars": len(prompt_str) if isinstance(prompt_str, str) else None,
        "promptLengthTokens": None,  # preenchido a partir do raw de resposta se disponível
    }

    # 5) Invoke Bedrock Runtime (medir latência)
    gen_cfg = payload.get("generationConfig", {}) or {}
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", current_app.config.get("DEFAULT_MAX_OUTPUT_TOKENS", 300))),
        "temperature": float(gen_cfg.get("temperature", current_app.config.get("DEFAULT_TEMPERATURE", 0.0))),
        "topP": float(gen_cfg.get("topP", current_app.config.get("DEFAULT_TOP_P", 0.1))),
    }

    model_id = current_app.config.get("BEDROCK_TURN_MODEL_ID")
    if not model_id:
        return jsonify({"error": "BEDROCK_TURN_MODEL_ID não configurado"}), 500

    start = time.time()
    try:
        # seu BedrockRuntimeService aceita prompt como string ou dict
        # passamos um dict com system vazio e user=prompt_str se preferir
        llm = _rt.invoke_text_model(model_id=model_id, prompt=prompt_str, generation=generation)
    except Exception as e:
        return jsonify({"error": f"Falha ao invocar modelo: {str(e)}"}), 502
    end = time.time()
    latency_ms = int((end - start) * 1000)

    # 6) Extrair telemetria do raw retornado
    raw = llm.get("raw", {}) or {}
    usage = _extract_usage_from_raw(raw)
    # preencher promptLengthTokens se runtime devolveu (alguns runtimes informam)
    # procurar nomes possíveis
    try:
        if isinstance(raw, dict):
            # algumas APIs retornam usage.promptTokens / usage.inputTokens etc.
            maybe_prompt_tokens = None
            u = raw.get("usage") or {}
            maybe_prompt_tokens = u.get("promptTokens") or u.get("inputTokens") or u.get("prompt_token_count")
            if maybe_prompt_tokens is not None:
                try:
                    prompt_metadata["promptLengthTokens"] = int(maybe_prompt_tokens)
                except Exception:
                    prompt_metadata["promptLengthTokens"] = None
    except Exception:
        prompt_metadata["promptLengthTokens"] = None

    # Tentativa de estimativa de custo (opcional) - usa custo por 1000 tokens do config se existir
    # variavel mantida em config.py = COST_PER_1000_TOKENS_USD
    cost_estimate_usd = None
    try:
        cost_per_1000 = float(current_app.config.get("COST_PER_1000_TOKENS_USD", 0.0))
        if usage.get("totalTokens") is not None and cost_per_1000 > 0:
            cost_estimate_usd = (usage["totalTokens"] / 1000.0) * cost_per_1000
    except Exception:
        cost_estimate_usd = None

    # 7) Compor retorno completo para o backend da aplicação
    response = {
        "replyText": llm.get("replyText", ""),
        "model": llm.get("modelId", model_id),
        "rawModelResponse": raw,  # para auditoria - guarde na DB se necessário
        "execution": {
            "latencyMs": latency_ms,
            "tokensIn": usage.get("tokensIn"),
            "tokensOut": usage.get("tokensOut"),
            "totalTokens": usage.get("totalTokens"),
            "costEstimateUsd": cost_estimate_usd,
        },
        "promptMetadata": prompt_metadata,
        "generationConfig": generation,
        "rag": rag_metrics,
        "telemetry": {
            "blueprintId": blueprint_id,
            "blueprintVersion": blueprint_version,
            "ragTopK": int(retrieval_cfg.get("topK", current_app.config.get("DEFAULT_TOP_K", 3))),
            "ragThreshold": float(retrieval_cfg.get("scoreThreshold", current_app.config.get("DEFAULT_SCORE_THRESHOLD", 0.1))),
            "usedKb": bool(kb_id and _kb),
        },
    }

    # Manter compatibilidade com o formato anterior (citations / flags top-level)
    response["citations"] = rag_metrics.get("citations", [])
    response["flags"] = rag_metrics.get("flags", [])

    return jsonify(response), 200