# External Benchmark Summary

| Benchmark | Method | Metric A | Metric B | Notes |
|---|---|---:|---:|---|
| ContractNLI | PolicyLLM | 0.4800 | 0.3890 | accuracy/macro_f1 |
| ContractNLI | Vanilla LLM | 0.8400 | 0.8402 | accuracy/macro_f1 |
| ContractNLI | System Prompt | 0.9200 | 0.9204 | accuracy/macro_f1 |
| LegalBench::unfair_tos | PolicyLLM | 0.4600 | 0.4580 | accuracy/macro_f1 |
| LegalBench::unfair_tos | Vanilla LLM | 0.5600 | 0.5098 | accuracy/macro_f1 |
| LegalBench::privacy_policy_entailment | PolicyLLM | 0.9800 | 0.9800 | accuracy/macro_f1 |
| LegalBench::privacy_policy_entailment | Vanilla LLM | 0.8800 | 0.8782 | accuracy/macro_f1 |
| CUAD | PolicyLLM | 0.7714 | 0.3222 | precision/recall |
| CUAD | Keyword + Rules | 0.5000 | 0.0556 | precision/recall |
| CUAD | RAG Extractor | 0.5000 | 0.0778 | precision/recall |

## Execution Status

- `contract_nli`: **ok**
- `legalbench_unfair_tos`: **ok**
- `legalbench_privacy_policy_entailment`: **ok**
- `cuad`: **ok**
