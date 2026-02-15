# Claude Bedrock Integration - Setup Complete

## ✅ Pipeline Verification

The generation pipeline has been verified to work correctly:

1. **Constitution Generation**: Generated first and stored in `stage1_explicit/ground_truth_constitution.json`
2. **Constitution Distribution**: Copied to all stage directories (stage1-4)
3. **Stage-Specific Document Generation**: Each stage uses the shared constitution with different prompt styles:
   - Stage 1: Explicit policy rewrites
   - Stage 2: Conflict-focused interpretations
   - Stage 3: Implicit/indirect interpretations
   - Stage 4: Mixed styles with sensitivity analysis
4. **Query Generation**: Test queries generated from constitution

## 🔑 Setup Files

- **bedrock_env/**: Isolated Python environment with all dependencies
  - anthropic SDK
  - requests
  - boto3
  - python-dotenv

- **.env**: Contains AWS_BEDROCK_API_KEY (not committed to git)

- **bedrock_client.py**: Claude client using Anthropic API
  - Handles model initialization
  - Prompt generation with system context
  - Output parsing (first line extraction)

- **launch_bedrock.sh**: Automated tmux launcher for full pipeline

## ⚠️ Authentication Issue

**Current Status**: API key authentication failed with error "invalid x-api-key"

**Possible Causes**:
1. The provided API key may need to be decoded differently
2. The key format may not be compatible with Anthropic v1 API
3. AWS Bedrock may require different authentication

**Next Steps**:
1. Verify the API key format / provide a new key
2. Update .env file with correct credentials
3. Re-run pipeline: `cd generation_scripts && source bedrock_env/bin/activate && python generate_dataset.py --root .. --use-bedrock`

## 📋 Pipeline Command

To run the full generation with Bedrock (once API key is verified):

```bash
cd /scratch2/f004ndc/ConstitutionCreator/synthetic_data/generation_scripts
source bedrock_env/bin/activate
python generate_dataset.py \
  --root .. \
  --num-policies 20 \
  --num-documents 20 \
  --num-queries 50 \
  --seed 42 \
  --use-bedrock
```

Or use the tmux launcher:
```bash
./launch_bedrock.sh
```

## 📁 Generated Output Structure

```
synthetic_data/
 stage1_explicit/
   ├── ground_truth_constitution.json
   └── documents/
 stage2_conflicts/
   ├── ground_truth_constitution.json
   └── documents/
 stage3_implicit/
   ├── ground_truth_constitution.json
   └── documents/
 stage4_mixed/
    ├── ground_truth_constitution.json
    ├── documents/
    ├── test_queries.json
    └── sensitivity/
        ├── baseline/seed_42/documents
        ├── explicit_heavy/seed_42/documents
        ├── implicit_heavy/seed_42/documents
        └── conflict_stress/seed_42/documents
```

