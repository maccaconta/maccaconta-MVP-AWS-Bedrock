# =============================================================================
# Arquivo: utils/validation_utils.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Validações técnicas mínimas de payload para garantir contrato do serviço.
# =============================================================================

from typing import Dict, Any, Optional


def validate_turn_payload(payload: Dict[str, Any]) -> Optional[str]:
    """
    Valida contrato mínimo do endpoint /turn.
    Retorna mensagem de erro (string) ou None quando ok.
    """
    required = ["sessionId", "turnId", "turnIndex", "userText", "contextPackage", "promptRef"]
    missing = [k for k in required if k not in payload]
    if missing:
        return f"Campos obrigatórios ausentes: {missing}"

    if not isinstance(payload.get("userText"), str) or not payload["userText"].strip():
        return "userText deve ser string não vazia"

    cp = payload.get("contextPackage", {})
    for k in ["conversationSummary", "lastTurns", "scenarioContext"]:
        if k not in cp:
            return f"contextPackage.{k} é obrigatório"

    if not isinstance(cp.get("lastTurns"), list):
        return "contextPackage.lastTurns deve ser um array"

    # promptRef aponta para um blueprint versionado no diretório templates/blueprints
    pr = payload.get("promptRef", {})
    for k in ["blueprintId", "blueprintVersion"]:
        if k not in pr or not pr.get(k):
            return f"promptRef.{k} é obrigatório"

    return None
