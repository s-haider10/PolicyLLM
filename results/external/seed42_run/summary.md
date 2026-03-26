# External Benchmark Summary

| Benchmark | Method | Metric A | Metric B | Notes |
|---|---|---:|---:|---|
| ContractNLI | PolicyLLM | 0.4800 | 0.3972 | accuracy/macro_f1 |
| ContractNLI | Vanilla LLM | 0.7800 | 0.7777 | accuracy/macro_f1 |
| ContractNLI | System Prompt | 0.8000 | 0.7995 | accuracy/macro_f1 |
| LegalBench::unfair_tos | PolicyLLM | 0.3600 | 0.3056 | accuracy/macro_f1 |
| LegalBench::unfair_tos | Vanilla LLM | 0.5800 | 0.5385 | accuracy/macro_f1 |
| LegalBench::privacy_policy_entailment | PolicyLLM | 0.9600 | 0.9599 | accuracy/macro_f1 |
| LegalBench::privacy_policy_entailment | Vanilla LLM | 0.7800 | 0.7688 | accuracy/macro_f1 |
| CUAD | PolicyLLM | 0.7550 | 0.2667 | precision/recall |
| CUAD | Keyword + Rules | 0.2000 | 0.0222 | precision/recall |
| CUAD | RAG Extractor | 0.3000 | 0.0556 | precision/recall |

## Execution Status

- `contract_nli`: **ok**
- `legalbench_unfair_tos`: **ok**
- `legalbench_privacy_policy_entailment`: **ok**
- `cuad`: **ok**
