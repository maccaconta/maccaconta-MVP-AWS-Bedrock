# endpoints/evaluate_endpoint.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify, current_app

from utils.prompt_repository import PromptRepository
from utils.prompt_utils import compose_prompt  # se existir; try/except é usado abaixo
from services.bedrock_runtime_service import BedrockRuntimeService
from services.bedrock_kb_service import BedrockKnowledgeBaseService

evaluate_bp = Blueprint("evaluate", __name__)

_repo: Optional[PromptRepository] = None
_rt: Optional[BedrockRuntimeService] = None
_kb: Optional[BedrockKnowledgeBaseService] = None


def _cfg(key: str, default: Any = None) -> Any:
    return current_app.config.get(key, default)


def _init() -> None:
    global _repo, _rt, _kb
    if _repo is None:
        _repo = PromptRepository(_cfg("TEMPLATES_ROOT"))
    if _rt is None:
        _rt = BedrockRuntimeService(region=_cfg("AWS_REGION"), timeout_seconds=int(_cfg("BEDROCK_TIMEOUT_SECONDS", 20)))
    if _kb is None and _cfg("BEDROCK_KB_ID"):
        _kb = BedrockKnowledgeBaseService(region=_cfg("AWS_REGION"), timeout_seconds=int(_cfg("BEDROCK_TIMEOUT_SECONDS", 20)))


def _extract_text_from_raw(raw: Dict[str, Any]) -> str:
    try:
        return raw.get("output", {}).get("message", {}).get("content", [])[0].get("text", "") or ""
    except Exception:
        return ""


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # tenta extrair JSON embutido
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                return None
    return None


def _collect_kb_evidence(kb_service: BedrockKnowledgeBaseService, kb_id: str, queries: List[str], top_k: int = 3) -> Dict[str, Any]:
    all_evidences: List[Dict[str, Any]] = []
    all_citations: List[str] = []
    for q in queries:
        try:
            resp = kb_service.retrieve(knowledge_base_id=kb_id, query_text=q, top_k=top_k)
            norm = BedrockKnowledgeBaseService.normalize(resp, score_threshold=float(_cfg("DEFAULT_SCORE_THRESHOLD", 0.1)))
            for ev in norm.get("evidences", []):
                all_evidences.append(ev)
            for c in norm.get("citations", []):
                all_citations.append(c)
        except Exception:
            continue

    seen = set()
    deduped = []
    for c in all_citations:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return {"evidences": all_evidences, "citations": deduped}


def _load_evaluate_template(template_file: str) -> Dict[str, Any]:
    """
    Tenta carregar o template por várias mecanicas:
      - se _repo expõe um loader compatível, tenta usar
      - caso contrário, lê o arquivo JSON diretamente de TEMPLATES_ROOT/evaluate/
    """
    # 1) tente usar PromptRepository se ele tiver algum loader conhecido
    global _repo
    if _repo is None:
        _repo = PromptRepository(_cfg("TEMPLATES_ROOT"))

    # tentativas com nomes prováveis de métodos
    loaders = ["load_template", "load_summarize_template", "load_blueprint", "load_component", "load_evaluate_template"]
    for m in loaders:
        if hasattr(_repo, m):
            try:
                func = getattr(_repo, m)
                # alguns loaders esperam (category, filename) outros só filename
                try:
                    return func("evaluate", template_file)  # tentativa 1
                except TypeError:
                    return func(template_file)  # tentativa 2
            except FileNotFoundError:
                raise
            except Exception:
                # se falhar, continua para próxima tentativa
                continue

    # fallback: ler arquivo diretamente
    templates_root = Path(_cfg("TEMPLATES_ROOT", Path.cwd() / "templates"))
    candidate = templates_root / "evaluate" / template_file
    if not candidate.exists():
        raise FileNotFoundError(f"Template não encontrado: {candidate}")
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler template {candidate}: {e}") from e


@evaluate_bp.route("/evaluate", methods=["POST"])
def post_evaluate():
    try:
        _init()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Payload JSON inválido"}), 400

    for k in ("sessionId", "rubricaId", "transcript"):
        if k not in payload:
            return jsonify({"error": f"Campo obrigatório ausente: {k}"}), 400

    session_id = payload["sessionId"]
    rubric_id = payload["rubricaId"]
    template_file = payload.get("templateFile", "avaliacao_rubrica_telemedicina_v1.json")
    transcript = payload["transcript"]
    ctx = transcript.get("context", {}) or {}
    turns: List[Dict[str, str]] = transcript.get("turns", []) or []

    # Carrega template com fallback robusto
    try:
        template = _load_evaluate_template(template_file)
    except FileNotFoundError as e:
        template = None
    except Exception as e:
        return jsonify({"error": f"Falha ao carregar template: {str(e)}"}), 500

    # RAG: coleta evidências se KB configurada
    kb_id = _cfg("BEDROCK_KB_ID")
    rag_context = {"evidences": [], "citations": []}
    if kb_id and _kb:
        queries = []
        prod = ctx.get("produto")
        if prod:
            queries.append(f"{prod} bula posologia fabricante farmacêutico responsável")
            queries.append(f"{prod} posologia indicação segurança gastrointestinais")
        for t in turns:
            text = t.get("text", "").strip()
            if len(text) > 10:
                queries.append(text)
        queries = list(dict.fromkeys([q for q in queries if q]))
        rag_context = _collect_kb_evidence(_kb, kb_id, queries, top_k=int(_cfg("DEFAULT_TOP_K", 3)))

    # Composição do prompt
    if template:
 
        try:
            prompt_out = compose_prompt(payload_json=payload, prompt_template_json=template, rag_context_json=rag_context)
            if isinstance(prompt_out, dict):
                prompt = {"system": prompt_out.get("system", ""), "user": prompt_out.get("user", "")}
            else:
                # string retornada ->  como user e pegar system do template
                system_text = (template.get("blocks", {}) or {}).get("system", "")
                prompt = {"system": system_text, "user": str(prompt_out)}
        except Exception:
            # fallback simples:   user com transcript + citations
            system_text = (template.get("blocks", {}) or {}).get("system", "") if template else ""
            user_lines = [f"Context: {json.dumps(ctx, ensure_ascii=False)}", "Transcript:"]
            for i, t in enumerate(turns, start=1):
                user_lines.append(f"{i}) {t.get('role')}: {t.get('text')}")
            if rag_context.get("citations"):
                user_lines.append("\nEvidências (citations):")
                user_lines += [f"- {c}" for c in rag_context.get("citations", [])]
            prompt = {"system": system_text, "user": "\n".join(user_lines)}
    else:
        # prompt minimal e prescritivo
        system_text = (
            "Você é um avaliador clínico-técnico. Sua tarefa é avaliar o desempenho do representante "
            "com base na transcrição e nas evidências listadas. Retorne SOMENTE um JSON válido com os campos: "
            "overallScore, categories, needsTraining, trainingFocus, citations, recommendations. Use as evidências citadas para justificar as notas."
        )
        user_lines = [f"Context: {json.dumps(ctx, ensure_ascii=False)}", "Transcript:"]
        for i, t in enumerate(turns, start=1):
            user_lines.append(f"{i}) {t.get('role')}: {t.get('text')}")
        if rag_context.get("citations"):
            user_lines.append("\nEvidências (citations):")
            user_lines += [f"- {c}" for c in rag_context.get("citations", [])]
        prompt = {"system": system_text, "user": "\n".join(user_lines)}

    # generation config
    gen_cfg = payload.get("generationConfig", {}) or {}
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", _cfg("DEFAULT_MAX_OUTPUT_TOKENS", 400))),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "topP": float(gen_cfg.get("topP", 0.1)),
    }

    model_id = _cfg("BEDROCK_EVALUATE_MODEL_ID")
    if not model_id:
        return jsonify({"error": "BEDROCK_EVALUATE_MODEL_ID não configurado"}), 500

    # chama runtime
    try:
        llm_resp = _rt.invoke_text_model(model_id=model_id, prompt=prompt, generation=generation)
    except Exception as e:
        return jsonify({"error": f"Falha ao invocar modelo: {str(e)}"}), 502

    raw = llm_resp.get("raw") or {}
    reply = llm_resp.get("replyText") or _extract_text_from_raw(raw)

    parsed = _try_parse_json(reply)
    if parsed and isinstance(parsed, dict):
        result = parsed
    else:
         return jsonify({"error": "Modelo não retornou JSON válido", "conteudo_bruto": reply, "raw": raw}), 502

    result.setdefault("sessionId", session_id)
    result.setdefault("model", model_id)
    result.setdefault("telemetry", {"generation": generation, "usedKb": bool(kb_id)})

    return jsonify(result), 200