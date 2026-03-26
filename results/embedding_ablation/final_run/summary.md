# Embedding Ablation Summary

| Method | Embedding | P-Rec | P-Prec | C-F1 | Comp. |
|---|---|---:|---:|---:|---:|
| RAG retrieval only | all-MiniLM-L6-v2 | 0.22 | 0.38 | 0.00 | 95.8 |
| RAG + Z3 hybrid | all-MiniLM-L6-v2 | 0.22 | 0.37 | 0.00 | 100.0 |
| RAG retrieval only | BAAI/bge-large-en-v1.5 | 0.09 | 0.22 | 0.00 | 95.8 |
| RAG + Z3 hybrid | BAAI/bge-large-en-v1.5 | 0.08 | 0.22 | 0.00 | 100.0 |
| RAG retrieval only | text-embedding-3-small | 0.13 | 0.30 | 0.00 | 100.0 |
| RAG + Z3 hybrid | text-embedding-3-small | 0.14 | 0.34 | 0.00 | 95.8 |
| PolicyLLM (Ours) | N/A | 0.00 | 0.00 | 0.00 | 0.0 |
