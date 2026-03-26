# External Benchmark Seed Variance (asserted_probe, full runs)

Seeds: 42, 43. `metric_a/metric_b` correspond to row notes (accuracy/macro_f1 or precision/recall).

| Benchmark | Method | Notes | metric_a mean±std | metric_b mean±std | Range a | Range b |
|---|---|---|---:|---:|---:|---:|
| CUAD | Keyword + Rules | precision/recall | 0.350 ± 0.150 | 0.039 ± 0.017 | [0.200, 0.500] | [0.022, 0.056] |
| CUAD | PolicyLLM | precision/recall | 0.763 ± 0.008 | 0.294 ± 0.028 | [0.755, 0.771] | [0.267, 0.322] |
| CUAD | RAG Extractor | precision/recall | 0.400 ± 0.100 | 0.067 ± 0.011 | [0.300, 0.500] | [0.056, 0.078] |
| ContractNLI | PolicyLLM | accuracy/macro_f1 | 0.480 ± 0.000 | 0.393 ± 0.004 | [0.480, 0.480] | [0.389, 0.397] |
| ContractNLI | System Prompt | accuracy/macro_f1 | 0.860 ± 0.060 | 0.860 ± 0.060 | [0.800, 0.920] | [0.799, 0.920] |
| ContractNLI | Vanilla LLM | accuracy/macro_f1 | 0.810 ± 0.030 | 0.809 ± 0.031 | [0.780, 0.840] | [0.778, 0.840] |
| LegalBench::privacy_policy_entailment | PolicyLLM | accuracy/macro_f1 | 0.970 ± 0.010 | 0.970 ± 0.010 | [0.960, 0.980] | [0.960, 0.980] |
| LegalBench::privacy_policy_entailment | Vanilla LLM | accuracy/macro_f1 | 0.830 ± 0.050 | 0.824 ± 0.055 | [0.780, 0.880] | [0.769, 0.878] |
| LegalBench::unfair_tos | PolicyLLM | accuracy/macro_f1 | 0.410 ± 0.050 | 0.382 ± 0.076 | [0.360, 0.460] | [0.306, 0.458] |
| LegalBench::unfair_tos | Vanilla LLM | accuracy/macro_f1 | 0.570 ± 0.010 | 0.524 ± 0.014 | [0.560, 0.580] | [0.510, 0.538] |
