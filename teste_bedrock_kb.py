# debug_kb_print.py
import json
from services.bedrock_kb_service import BedrockKnowledgeBaseService
import os

KB_ID = os.environ.get("BEDROCK_KB_ID", "QFFFVTP1SD")
QUERY = "Quem é Adriano Pinheiro Coelho?"   # substitua pela pergunta exata que fez
TOP_K = 100

svc = BedrockKnowledgeBaseService(region=os.environ.get("AWS_REGION","us-east-1"), timeout_seconds=30)
resp = svc.retrieve(knowledge_base_id=KB_ID, query_text=QUERY, top_k=TOP_K)

results = resp.get("retrievalResults", [])
print("TOTAL results:", len(results))
print("Raw response (resumo):")
print(json.dumps({k: resp.get(k) for k in resp.keys() if k!='retrievalResults'}, ensure_ascii=False, indent=2))

for i, r in enumerate(results):
    print("\n=== RESULT", i, "score=", r.get("score"))
    print(json.dumps(r, ensure_ascii=False, indent=2))