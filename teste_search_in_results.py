# search_in_results.py
import json
from services.bedrock_kb_service import BedrockKnowledgeBaseService
import os

KB_ID = os.environ.get("BEDROCK_KB_ID", "QFFFVTP1SD")
QUERY = "Quem é Adriano Pinheiro Coelho?"  # cole aqui o trecho exato que espera
TOP_K = 50

svc = BedrockKnowledgeBaseService(region=os.environ.get("AWS_REGION","us-east-1"), timeout_seconds=30)
resp = svc.retrieve(knowledge_base_id=KB_ID, query_text=QUERY, top_k=TOP_K)

results = resp.get("retrievalResults", []) or []
found = False
for i, r in enumerate(results):
    # tenta extrair conteúdo textualmente
    c = r.get("content", {}) or {}
    text = ""
    if isinstance(c, dict):
        text = c.get("text") or c.get("body") or ""
    else:
        text = str(c)
    if QUERY in text:
        print(f"ENCONTREI em result {i} score={r.get('score')}")
        print("location:", r.get("location"))
        print("metadata:", r.get("metadata"))
        print("snippet (preview):", text[:1000])
        found = True
        break

if not found:
    print("Não encontrei a substring exata em nenhum retrievalResult. Mostrando primeiros 5 snippets para inspeção:")
    for i, r in enumerate(results[:5]):
        print("\n--- RESULT", i, "score=", r.get("score"))
        print("location:", r.get("location"))
        print("metadata:", r.get("metadata"))
        c = r.get("content", {}) or {}
        if isinstance(c, dict):
            print("content.text preview:", (c.get("text") or c.get("body") or "")[:1000])
        else:
            print("content preview:", str(c)[:1000])