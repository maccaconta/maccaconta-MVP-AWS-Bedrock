# utils/prompt_utils.py
from typing import Optional, Dict, Any, List


def compose_prompt(
    payload_json: Dict[str, Any],
    prompt_template_json: Dict[str, Any],
    rag_context_json: Optional[Dict[str, Any]] = None,
    extra_sections_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compoe o prompt final a partir de múltiplas fontes em JSON.

    Entradas principais:
    - payload_json: JSON recebido pelo endpoint (contém userText, conversationSummary, lastTurns, etc).
    - prompt_template_json: JSON do template versionado (system, policy, script, output, etc).
    - rag_context_json: JSON retornado pelo motor de RAG (evidências, flags, citations).
    - extra_sections_json: JSON opcional com outras seções (ex: metadados adicionais, instruções específicas).

    Saída:
    - String de prompt final, pronta para envio ao LLM.

    Observação:
    - Esta função NÃO chama Bedrock, apenas concatena e organiza as partes.
    """

    # 1) Extrai blocos do template (system, policy, script de cenário, etc.)
    system_block = prompt_template_json.get("system", "")
    policy_rules: List[str] = prompt_template_json.get("policyRules", [])
    script_guidelines: List[str] = prompt_template_json.get("scriptGuidelines", [])
    output_contract = prompt_template_json.get("outputContract", {})

    # 2) Extrai elementos de contexto do payload
    conversation_summary = payload_json.get("conversationSummary", "")
    last_turns = payload_json.get("lastTurns", [])
    user_text = payload_json.get("userText", "")

    # 3) Extrai evidências do RAG, se houver
    evidences = []
    no_evidence = False
    if rag_context_json:
        evidences = rag_context_json.get("evidences", [])
        no_evidence = rag_context_json.get("no_evidence", False)

    # 4) Monta partes de texto
    parts: List[str] = []

    # 4.1. Bloco SYSTEM
    if system_block:
        parts.append("SYSTEM:")
        parts.append(system_block)
        parts.append("")

    # 4.2. Bloco de POLICIES
    if policy_rules:
        parts.append("POLICIES:")
        for rule in policy_rules:
            parts.append(f"- {rule}")
        parts.append("")

    # 4.3. Bloco de SCRIPT/CENARIO
    if script_guidelines:
        parts.append("SCRIPT DE CONSULTA:")
        for guideline in script_guidelines:
            parts.append(f"- {guideline}")
        parts.append("")

    # 4.4. Bloco de CONTEXTO (summary + lastTurns)
    if conversation_summary:
        parts.append("RESUMO DA CONVERSA ATUAL:")
        parts.append(conversation_summary)
        parts.append("")

    if last_turns:
        parts.append("ULTIMOS TURNS:")
        for idx, t in enumerate(last_turns, start=1):
            inp = t.get("input", "")
            out = t.get("output", "")
            parts.append(f"- Turn {idx} (usuario): {inp}")
            parts.append(f"- Turn {idx} (assistente): {out}")
        parts.append("")

    # 4.5. Bloco de EVIDENCIAS DO RAG
    if evidences:
        parts.append("EVIDENCIAS RELEVANTES (RAG):")
        for ev in evidences:
            doc_id = ev.get("docId", "")
            chunk_id = ev.get("chunkId", "")
            snippet = ev.get("snippet", "")
            parts.append(f"- [{doc_id} / {chunk_id}] {snippet}")
        parts.append("")

    # 4.6. Bloco de CONTRATO DE SAIDA
    if output_contract:
        parts.append("INSTRUCOES DE SAIDA:")
        if "format" in output_contract:
            parts.append(f"- Formato: {output_contract['format']}")
        if "maxSentences" in output_contract:
            parts.append(f"- Maximo de frases: {output_contract['maxSentences']}")
        parts.append("")

    # 4.7. Bloco EXTRA (quando existir)
    if extra_sections_json:
        parts.append("INSTRUCOES EXTRAS:")
        # Aqui você pode definir um formato livre para extras
        parts.append(str(extra_sections_json))
        parts.append("")

    # 4.8. Bloco final de INPUT DO USUARIO
    parts.append("USUARIO:")
    parts.append(user_text)
    parts.append("")
    parts.append("ASSISTENTE:")

    # 5) Junta todas as partes em uma única string com quebras de linha
    prompt_str = "\n".join(parts)

    return prompt_str
