# Test Data — Dataset Documentation

## Overview

This directory contains policy documents used for evaluating PolicyLLM. All documents are publicly available and used under fair use for research purposes.

## Documents

| File | Source | License / Terms | Description |
|------|--------|-----------------|-------------|
| `zara_terms and conditions.pdf` | [zara.com](https://www.zara.com/) | Public terms of service | Retail terms covering returns, refunds, delivery, user conduct, IP, and dispute resolution |
| `E6_R2_Addendum.pdf` | US Federal Government | Public domain (US law) | Health Insurance Portability and Accountability Act (HIPAA) — privacy, security, penalties |
| `NICE*.pdf` | [NICE (UK)](https://www.nice.org.uk/) | Open Government Licence | National Institute for Health and Care Excellence guidelines |
| `WTO*.pdf` | [WTO](https://www.wto.org/) | Public domain (international treaty) | World Trade Organization agreement text |
| `yc_safe*.pdf` | [Y Combinator](https://www.ycombinator.com/documents) | Public template | Simple Agreement for Future Equity (SAFE) template |

## Domains Covered

| Domain | Documents | Example Policies |
|--------|-----------|-----------------|
| Refund | Zara T&C | Return windows, receipt requirements, prohibited returns |
| Privacy | HIPAA | PHI protection, disclosure restrictions, identifiers |
| Security | HIPAA, Zara T&C | Network security, access controls, safeguards |
| HR | NICE | Employee conduct, grievance procedures |
| Escalation | WTO, Zara T&C | Dispute resolution, arbitration, appeals |
| Finance | YC SAFE | Investment terms, conversion triggers |

## Ground Truth Annotations

Expert-annotated reference data for evaluation is stored in `eval/reference_data/`:

- `extraction_gt.json` — Per-document expected policies, conditions, and conflicts
- `enforcement_gt.json` — Test queries with expected enforcement decisions

## Usage

```bash
# Process all test PDFs
python run_extract_tests_pdfs.py

# Process a single PDF
python main.py extract tests/zara_terms\ and\ conditions.pdf --out results/zara/

# Run evaluation against ground truth
python run_extract_tests_pdfs.py --eval-only
```

## Citation

If you use these documents in your research, please cite the original sources:

- **HIPAA**: Health Insurance Portability and Accountability Act of 1996, Pub.L. 104–191.
- **NICE**: National Institute for Health and Care Excellence, UK.
- **WTO**: World Trade Organization Agreement texts.
- **YC SAFE**: Y Combinator, "Simple Agreement for Future Equity," 2018.
