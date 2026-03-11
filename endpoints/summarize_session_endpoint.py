# =============================================================================
# endpoints/summarize_session_endpoint.py
# Endpoint: POST /v1/ai/summarize-session
#
# Objetivo:
#   Gerar/atualizar um resumo clínico da sessão a partir de:
#     - currentSummary (sumário acumulado)
#     - recentTurns (lista de turns: input/output)
#   O endpoint:
#     1) Valida payload
#     2) Carrega template de prompt (se houver)
#     3) (Opcional) usa RAG para buscar evidências (opcional)
#     4) Monta prompt com instruções de sistema (fortes) para forçar JSON com summary
#     5) Invoca Bedrock Runtime (modelo configurável)
#     6) Extrai texto do retorno e tenta parse para JSON (se o modelo obedecer)
#     7) Retorna newSummary (texto) e raw (payload do modelo) + telemetria
#
# Observações de design:
#   - RAG é opcional: se BEDROCK_KB_ID não estiver configurado, segue sem evidências.
#   - Uso de generationConfig com defaults para comportamento determinístico em resumos.
#   - Extração defensiva do texto retornado pelo modelo (suporta várias formas de resposta).
# =============================================================================
# endpoints/summarize_session_endpoint.py
# endpoints/summarize_session_endpoint.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify, current_app

from utils.prompt_repository import PromptRepository
from services.bedrock_runtime_service import BedrockRuntimeService

summarize_bp = Blueprint("summarize", __name__)

_repo: Optional[PromptRepository] = None
_rt: Optional[BedrockRuntimeService] = None


def _cfg(key: str, default: Any = None) -> Any:
    return current_app.config.get(key, default)


def _init() -> None:
    global _repo, _rt
    if _repo is None:
        _repo = PromptRepository(current_app.config["TEMPLATES_ROOT"])
    if _rt is None:
        _rt = BedrockRuntimeService(
            region=current_app.config["AWS_REGION"],
            timeout_seconds=int(current_app.config.get("BEDROCK_TIMEOUT_SECONDS", 20)),
        )


def _build_system_from_template(tpl: Dict[str, Any]) -> str:
    """
    Monta instruções de SYSTEM a partir do template que você mostrou:
    {
      "meta": {...},
      "blocks": {"system": "...", "policy": [...], "outputContract": {...}}
    }
    """
    blocks = tpl.get("blocks", {}) or {}
    system = (blocks.get("system") or "").strip()
    policy = blocks.get("policy") or []
    output_contract = blocks.get("outputContract") or {}

    parts: List[str] = []
    if system:
        parts.append(system)

    if policy:
        parts.append("\nPOLÍTICAS:")
        for p in policy:
            parts.append(f"- {p}")

    if output_contract:
        fmt = output_contract.get("format", "texto")
        tamanho = output_contract.get("tamanho_maximo", "medio")
        parts.append("\nCONTRATO DE SAÍDA:")
        parts.append(f"- Formato: {fmt}")
        parts.append(f"- Tamanho máximo: {tamanho}")

    # Reforço anti-alucinação e anti-perguntas
    parts.append("\nREGRAS ADICIONAIS:")
    parts.append("- Responda SOMENTE com o resumo; não faça perguntas.")
    parts.append("- Não invente nomes, doses ou dados que não estejam no diálogo.")
    parts.append("- Se algo não foi mencionado, omita.")

    return "\n".join(parts).strip()


def _build_user_from_payload(payload: Dict[str, Any]) -> str:
    current_summary = (payload.get("currentSummary") or "").strip()
    recent_turns: List[Dict[str, str]] = payload.get("recentTurns") or []

    lines: List[str] = []
    lines.append("RESUMO ATUAL (se houver):")
    lines.append(current_summary if current_summary else "(vazio)")
    lines.append("")
    lines.append("DIÁLOGO DA SESSÃO (recentTurns):")

    for i, t in enumerate(recent_turns, start=1):
        inp = (t.get("input") or "").strip()
        out = (t.get("output") or "").strip()
        lines.append(f"{i}) USER: {inp}")
        lines.append(f"   ASSISTANT: {out}")

    lines.append("")
    lines.append("TAREFA: Gere o resumo técnico e conciso desta sessão, baseado SOMENTE no que está acima.")
    return "\n".join(lines)


def _try_extract_summary(text: str, expected_format: str) -> Dict[str, Any]:
    """
    Se o template disser 'format: json', tenta parsear. Caso contrário, retorna texto puro.
    """
    text = (text or "").strip()
    if expected_format.lower() != "json":
        return {"summary": text}

    # tenta JSON direto
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # tenta extrair substring JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # fallback
    return {"summary": text}


@summarize_bp.route("/summarize-session", methods=["POST"])
def post_summarize_session():
    _init()

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON inválido"}), 400

    for k in ["sessionId", "currentSummary", "recentTurns"]:
        if k not in payload:
            return jsonify({"error": f"Campo obrigatório ausente: {k}"}), 400

    template_file = payload.get("templateFile", "resumo_sessao_v1.json")
    try:
        template = _repo.load_summarize_template(template_file)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    system_text = _build_system_from_template(template)
    user_text = _build_user_from_payload(payload)

    model_id = current_app.config.get("BEDROCK_SUMMARIZE_MODEL_ID")
    if not model_id:
        return jsonify({"error": "BEDROCK_SUMMARIZE_MODEL_ID não configurado"}), 500

    # Defaults para reduzir “aleatoriedade” em resumo
    gen_cfg = payload.get("generationConfig", {}) or {}
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", 250)),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "topP": float(gen_cfg.get("topP", 0.1)),
    }

    llm = _rt.invoke_text_model(
        model_id=model_id,
        prompt={"system": system_text, "user": user_text},
        generation=generation,
    )

    raw = llm.get("raw") or {}
    reply_text = (llm.get("replyText") or "").strip()

    expected_format = ((template.get("blocks", {}) or {}).get("outputContract", {}) or {}).get("format", "texto")
    parsed = _try_extract_summary(reply_text, expected_format)

    # se format for texto, summary = reply_text
    new_summary = (parsed.get("summary") if isinstance(parsed, dict) else None) or reply_text

    return jsonify({
        "sessionId": payload.get("sessionId"),
        "model": llm.get("modelId", model_id),
        "newSummary": new_summary,
        "newSummaryRaw": raw,
        "telemetry": {
            "templateFile": template_file,
            "expectedFormat": expected_format,
            "generation": generation
        }
    }), 200