# test_retrieve.py
import os
from services.bedrock_kb_service import BedrockKnowledgeBaseService

def main():
    region = os.environ.get("AWS_REGION", "us-east-1")
    timeout = int(os.environ.get("BEDROCK_TIMEOUT_SECONDS", "10"))
    kb_id = os.environ.get("KB_ID", "BJWEDBPNJH")  # ou substitua direto

    svc = BedrockKnowledgeBaseService(region=region, timeout_seconds=timeout)
    try:
        resp = svc.retrieve(knowledge_base_id=kb_id, query_text="Sintomas de dor torácica", top_k=3)
        norm = svc.normalize(resp, score_threshold=0.1)
        print("Normalize result:", norm)
    except Exception as e:
        print("Erro ao chamar KB:", type(e).__name__, e)

if __name__ == "__main__":
    main()