# RAGMaintenanceArchitecture

MRGA(Multi-agent RAG + Guard + Memory) baseline for ATE maintenance diagnosis.

## Architecture
- Python API Layer
- Ouroboros-style Workflow Engine
  - ContextAgent
  - RetrievalAgent
  - DiagnosisAgent
  - ProcedureAgent
  - TrustGateAgent
  - FeedbackAgent
  - MemoryAgent
- RAG Layer
  - Chroma-like retriever (pluggable)
  - ICD/절차서/정비이력/시험로그/FMEA documents
- Persistent Memory
  - 작업자 피드백
  - 실제 원인
  - 조치 결과
  - 반복 장애 패턴

## Quick start
```bash
python -m pip install -e .
python -m mrga.main --query "GPRA 주파수변환 FAIL 발생"
pytest -q
```

## Notes
- This is a deterministic baseline implementation to stabilize behavior first.
- Chroma DB can be connected later by replacing `KeywordRetriever` with a vector retriever adapter.
