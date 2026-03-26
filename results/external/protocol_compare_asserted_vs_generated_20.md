# Protocol Comparison: asserted_probe vs generated_response (matched 20-example slices)

All rows use identical split IDs and seed=42; deltas are `generated - asserted`.

| Benchmark | Method | Asserted acc | Generated acc | Δ acc | Asserted macro-F1 | Generated macro-F1 | Δ macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ContractNLI | PolicyLLM | 0.400 | 0.300 | -0.100 | 0.315 | 0.154 | -0.161 |
| ContractNLI | System Prompt | 0.900 | 0.850 | -0.050 | 0.897 | 0.850 | -0.047 |
| ContractNLI | Vanilla LLM | 0.850 | 0.850 | +0.000 | 0.847 | 0.847 | +0.000 |
| LegalBench::privacy_policy_entailment | PolicyLLM | 0.950 | 0.500 | -0.450 | 0.950 | 0.451 | -0.499 |
| LegalBench::privacy_policy_entailment | Vanilla LLM | 0.800 | 0.800 | +0.000 | 0.792 | 0.792 | +0.000 |
| LegalBench::unfair_tos | PolicyLLM | 0.400 | 0.500 | +0.100 | 0.286 | 0.333 | +0.048 |
| LegalBench::unfair_tos | Vanilla LLM | 0.600 | 0.600 | +0.000 | 0.524 | 0.524 | +0.000 |
