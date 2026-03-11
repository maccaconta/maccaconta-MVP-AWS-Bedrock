# =============================================================================
# Arquivo: utils/prompt_composer.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Composição do prompt final do Turn a partir de:
#   - Payload do endpoint (contextPackage)
#   - Blueprint (receita)
#   - Componentes (persona, especialidade, cenário, políticas, contrato de saída)
#   - Evidências do RAG (Knowledge Base / S3 Vectors)
# =============================================================================

from typing import Dict, Any, List, Optional


def compose_turn_prompt(
    payload_json: Dict[str, Any],
    blueprint_json: Dict[str, Any],
    components_json: Dict[str, Dict[str, Any]],
    rag_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Monta o prompt final (string) para o endpoint /turn.

    components_json deve conter:
      - persona
      - especialidade
      - cenario
      - politicas
      - saida
    """
    context_package = payload_json["contextPackage"]
    user_text = payload_json["userText"]
    conversation_summary = context_package.get("conversationSummary", "")
    last_turns = context_package.get("lastTurns", [])

    persona = components_json["persona"]
    especialidade = components_json["especialidade"]
    cenario = components_json["cenario"]
    politicas = components_json["politicas"]
    saida = components_json["saida"]

    evidences = []
    flags: List[str] = []
    if rag_json:
        evidences = rag_json.get("evidences", [])
        flags = rag_json.get("flags", [])

    parts: List[str] = []

    # 1) Cabeçalho do prompt (identificação operacional)
    parts.append("## CONTEXTO OPERACIONAL")
    parts.append(f"- sessionId: {payload_json.get('sessionId')}")
    parts.append(f"- turnId: {payload_json.get('turnId')}")
    parts.append(f"- turnIndex: {payload_json.get('turnIndex')}")
    parts.append("")

    # 2) Instruções de comportamento (persona)
    parts.append("## PERSONA (COMPORTAMENTO)")
    parts.append(persona["conteudo"]["descricao_curta"])
    for regra in persona["conteudo"].get("regras", []):
        parts.append(f"- {regra}")
    parts.append("")

    # 3) Contexto da especialidade (domínio)
    parts.append("## ESPECIALIDADE (CONTEXTO CLINICO)")
    parts.append(especialidade["conteudo"]["descricao_curta"])
    for ponto in especialidade["conteudo"].get("pontos_chave", []):
        parts.append(f"- {ponto}")
    parts.append("")

    # 4) Cenário/script (roteiro do atendimento)
    parts.append("## CENARIO (ROTEIRO DE CONSULTA)")
    parts.append(cenario["conteudo"]["descricao_curta"])
    for passo in cenario["conteudo"].get("passos", []):
        parts.append(f"- {passo}")
    parts.append("")

    # 5) Políticas (restrições e governança)
    parts.append("## POLITICAS (REGRAS DE SEGURANCA E QUALIDADE)")
    for regra in politicas["conteudo"].get("regras", []):
        parts.append(f"- {regra}")
    parts.append("")

    # 6) Context package (continuidade conversacional controlada)
    parts.append("## RESUMO DA SESSAO (conversationSummary)")
    parts.append(conversation_summary.strip() if conversation_summary else "(vazio)")
    parts.append("")

    parts.append("## ULTIMAS INTERACOES (lastTurns)")
    if last_turns:
        for i, t in enumerate(last_turns, start=1):
            parts.append(f"- {i}. usuario: {t.get('input','')}")
            parts.append(f"     assistente: {t.get('output','')}")
    else:
        parts.append("(vazio)")
    parts.append("")

    # 7) Evidências do RAG (se houver)
    parts.append("## EVIDENCIAS (RAG - KB / S3 VECTORS)")
    if evidences:
        for ev in evidences:
            parts.append(
                f"- fonte={ev.get('docId')} trecho={ev.get('chunkId')} score={ev.get('score')}\n"
                f"  texto={ev.get('snippet')}"
            )
    else:
        parts.append("(nenhuma evidencia retornada)")
    parts.append("")

    # 8) Contrato de saída (formato e tamanho)
    parts.append("## CONTRATO DE SAIDA")
    contrato = saida["conteudo"]
    parts.append(f"- formato: {contrato.get('formato','texto')}")
    parts.append(f"- tamanho_maximo: {contrato.get('tamanho_maximo','curto')}")
    parts.append(f"- limite_frases: {contrato.get('limite_frases', 3)}")
    parts.append("")

    # 9) Input do usuário
    parts.append("## ENTRADA DO USUARIO")
    parts.append(user_text.strip())
    parts.append("")
    parts.append("## RESPOSTA DO ASSISTENTE")
    parts.append("Responda agora seguindo rigorosamente as instrucoes acima.")

    return "\n".join(parts)
