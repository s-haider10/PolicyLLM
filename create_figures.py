#!/usr/bin/env python3
"""
Create figures analyzing extraction pipeline performance across stages.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Paths
PROJECT_ROOT = Path("/scratch2/f004ndc/ConstitutionCreator")
OUTPUT_DIR = PROJECT_ROOT / "synthetic_tests/output"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load test summary
with open(OUTPUT_DIR / "pipeline_test_summary.json") as f:
    test_summary = json.load(f)

# Load ground truth for each stage
def load_ground_truth(stage):
    gt_path = PROJECT_ROOT / f"synthetic_data/{stage}/ground_truth_constitution.json"
    with open(gt_path) as f:
        return json.load(f)

# Load extracted policies
def load_extracted_policies(stage):
    policies_file = None
    for test_case in test_summary["test_cases"]:
        if test_case["stage"] == stage:
            policies_file = test_case["extraction"]["policies_file"]
            break
    
    if not policies_file or not Path(policies_file).exists():
        return []
    
    policies = []
    with open(policies_file) as f:
        for line in f:
            if line.strip():
                policies.append(json.loads(line))
    return policies

# Calculate reconstruction metrics
def calculate_metrics(stage, ground_truth, extracted):
    gt_policies = ground_truth["policies"]
    
    # Count metrics
    gt_count = len(gt_policies)
    extracted_count = len(extracted)
    
    # Domain distribution
    gt_domains = Counter([p["domain"] for p in gt_policies])
    extracted_domains = Counter([p.get("domain", "unknown") for p in extracted])
    
    # Condition metrics
    gt_conditions = sum(len(p.get("conditions", [])) for p in gt_policies)
    extracted_conditions = sum(len(p.get("conditions", [])) for p in extracted)
    
    # Action metrics
    gt_actions = sum(len(p.get("actions", [])) for p in gt_policies)
    extracted_actions = sum(len(p.get("actions", [])) for p in extracted)
    
    return {
        "stage": stage,
        "gt_policies": gt_count,
        "extracted_policies": extracted_count,
        "policy_recall": extracted_count / gt_count if gt_count > 0 else 0,
        "gt_domains": gt_domains,
        "extracted_domains": extracted_domains,
        "gt_conditions": gt_conditions,
        "extracted_conditions": extracted_conditions,
        "condition_recall": extracted_conditions / gt_conditions if gt_conditions > 0 else 0,
        "gt_actions": gt_actions,
        "extracted_actions": extracted_actions,
        "action_recall": extracted_actions / gt_actions if gt_actions > 0 else 0,
    }

# Collect metrics for all stages
stages = ["stage1_explicit", "stage2_conflicts", "stage3_implicit", "stage4_mixed"]
stage_labels = ["Stage 1\nExplicit", "Stage 2\nConflicts", "Stage 3\nImplicit", "Stage 4\nMixed"]
metrics = []

for stage in stages:
    gt = load_ground_truth(stage)
    extracted = load_extracted_policies(stage)
    m = calculate_metrics(stage, gt, extracted)
    metrics.append(m)
    print(f"{stage}: {m['extracted_policies']}/{m['gt_policies']} policies extracted")

# =============================================================================
# FIGURE 1: Reconstruction Accuracy Overview
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Extraction Pipeline: Reconstruction Accuracy Across Stages", fontsize=16, fontweight='bold')

# 1.1: Policy Count Accuracy
ax = axes[0, 0]
x = np.arange(len(stages))
width = 0.35

gt_counts = [m["gt_policies"] for m in metrics]
extracted_counts = [m["extracted_policies"] for m in metrics]

bars1 = ax.bar(x - width/2, gt_counts, width, label='Ground Truth', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, extracted_counts, width, label='Extracted', color='#3498db', alpha=0.8)

ax.set_ylabel('Policy Count', fontweight='bold')
ax.set_title('Policy Count: Ground Truth vs Extracted', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stage_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1.2: Recall Rates
ax = axes[0, 1]
recall_rates = [m["policy_recall"] * 100 for m in metrics]
colors = ['#e74c3c' if r < 50 else '#f39c12' if r < 80 else '#2ecc71' for r in recall_rates]

bars = ax.bar(stage_labels, recall_rates, color=colors, alpha=0.8)
ax.set_ylabel('Recall Rate (%)', fontweight='bold')
ax.set_title('Policy Recall Rate by Stage', fontweight='bold')
ax.set_ylim([0, 110])
ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Recall')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1.3: Condition Recall
ax = axes[1, 0]
condition_recall_rates = [m["condition_recall"] * 100 for m in metrics]
gt_conditions = [m["gt_conditions"] for m in metrics]
extracted_conditions = [m["extracted_conditions"] for m in metrics]

x = np.arange(len(stages))
bars1 = ax.bar(x - width/2, gt_conditions, width, label='Ground Truth', color='#9b59b6', alpha=0.8)
bars2 = ax.bar(x + width/2, extracted_conditions, width, label='Extracted', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Condition Count', fontweight='bold')
ax.set_title('Condition Extraction: Ground Truth vs Extracted', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stage_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 1.4: Action Recall
ax = axes[1, 1]
action_recall_rates = [m["action_recall"] * 100 for m in metrics]
gt_actions = [m["gt_actions"] for m in metrics]
extracted_actions = [m["extracted_actions"] for m in metrics]

bars1 = ax.bar(x - width/2, gt_actions, width, label='Ground Truth', color='#f39c12', alpha=0.8)
bars2 = ax.bar(x + width/2, extracted_actions, width, label='Extracted', color='#1abc9c', alpha=0.8)

ax.set_ylabel('Action Count', fontweight='bold')
ax.set_title('Action Extraction: Ground Truth vs Extracted', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stage_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "reconstruction_accuracy_overview.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/reconstruction_accuracy_overview.png")
plt.close()

# =============================================================================
# FIGURE 2: Domain Coverage Analysis
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Domain Distribution: Ground Truth vs Extracted Policies", fontsize=16, fontweight='bold')

for idx, (stage, stage_label) in enumerate(zip(stages, stage_labels)):
    ax = axes[idx // 2, idx % 2]
    m = metrics[idx]
    
    # Get all domains
    all_domains = set(m["gt_domains"].keys()) | set(m["extracted_domains"].keys())
    domains = sorted(all_domains)
    
    x = np.arange(len(domains))
    gt_values = [m["gt_domains"].get(d, 0) for d in domains]
    extracted_values = [m["extracted_domains"].get(d, 0) for d in domains]
    
    bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, extracted_values, width, label='Extracted', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Policy Count', fontweight='bold')
    ax.set_title(f'{stage_label}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "domain_coverage_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/domain_coverage_analysis.png")
plt.close()

# =============================================================================
# FIGURE 3: Stage 4 Sensitivity Analysis
# =============================================================================
# For stage 4 (mixed), analyze the diversity of policy characteristics

stage4_gt = load_ground_truth("stage4_mixed")
stage4_extracted = load_extracted_policies("stage4_mixed")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Stage 4 (Mixed): Sensitivity Analysis", fontsize=16, fontweight='bold')

# 3.1: Policy Complexity Distribution
ax = axes[0, 0]
gt_complexity = [len(p.get("conditions", [])) + len(p.get("actions", [])) for p in stage4_gt["policies"]]
extracted_complexity = [len(p.get("conditions", [])) + len(p.get("actions", [])) for p in stage4_extracted]

bins = np.arange(0, max(max(gt_complexity), max(extracted_complexity or [0])) + 2)
ax.hist([gt_complexity, extracted_complexity], bins=bins, label=['Ground Truth', 'Extracted'],
        color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Complexity (Conditions + Actions)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Policy Complexity Distribution', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3.2: Exception Handling
ax = axes[0, 1]
gt_with_exceptions = sum(1 for p in stage4_gt["policies"] if p.get("exceptions"))
gt_without_exceptions = len(stage4_gt["policies"]) - gt_with_exceptions
extracted_with_exceptions = sum(1 for p in stage4_extracted if p.get("exceptions"))
extracted_without_exceptions = len(stage4_extracted) - extracted_with_exceptions

x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [gt_with_exceptions, gt_without_exceptions], width, 
               label='Ground Truth', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, [extracted_with_exceptions, extracted_without_exceptions], width,
               label='Extracted', color='#f39c12', alpha=0.8)

ax.set_ylabel('Policy Count', fontweight='bold')
ax.set_title('Exception Handling Coverage', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['With Exceptions', 'Without Exceptions'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3.3: Operator Diversity
ax = axes[1, 0]
gt_operators = []
for p in stage4_gt["policies"]:
    for c in p.get("conditions", []):
        gt_operators.append(c.get("op", "unknown"))

extracted_operators = []
for p in stage4_extracted:
    for c in p.get("conditions", []):
        extracted_operators.append(c.get("operator", c.get("op", "unknown")))

gt_op_counts = Counter(gt_operators)
extracted_op_counts = Counter(extracted_operators)

all_ops = sorted(set(gt_op_counts.keys()) | set(extracted_op_counts.keys()))
x = np.arange(len(all_ops))
gt_values = [gt_op_counts.get(op, 0) for op in all_ops]
extracted_values = [extracted_op_counts.get(op, 0) for op in all_ops]

bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color='#9b59b6', alpha=0.8)
bars2 = ax.bar(x + width/2, extracted_values, width, label='Extracted', color='#1abc9c', alpha=0.8)

ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Condition Operator Diversity', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_ops, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3.4: Action Type Distribution
ax = axes[1, 1]
gt_action_types = []
for p in stage4_gt["policies"]:
    for a in p.get("actions", []):
        gt_action_types.append(a.get("type", "unknown"))

extracted_action_types = []
for p in stage4_extracted:
    for a in p.get("actions", []):
        extracted_action_types.append(a.get("type", "unknown"))

gt_action_counts = Counter(gt_action_types)
extracted_action_counts = Counter(extracted_action_types)

all_action_types = sorted(set(gt_action_counts.keys()) | set(extracted_action_counts.keys()))
x = np.arange(len(all_action_types))
gt_values = [gt_action_counts.get(t, 0) for t in all_action_types]
extracted_values = [extracted_action_counts.get(t, 0) for t in all_action_types]

bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color='#e67e22', alpha=0.8)
bars2 = ax.bar(x + width/2, extracted_values, width, label='Extracted', color='#34495e', alpha=0.8)

ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Action Type Distribution', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_action_types, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage4_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/stage4_sensitivity_analysis.png")
plt.close()

# =============================================================================
# FIGURE 4: Overall Metrics Summary Heatmap
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics_matrix = []
metric_names = []

for m in metrics:
    metrics_matrix.append([
        m["policy_recall"] * 100,
        m["condition_recall"] * 100,
        m["action_recall"] * 100,
    ])

metric_names = ["Policy Recall (%)", "Condition Recall (%)", "Action Recall (%)"]

# Create heatmap
im = ax.imshow(np.array(metrics_matrix).T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax.set_xticks(np.arange(len(stages)))
ax.set_yticks(np.arange(len(metric_names)))
ax.set_xticklabels(stage_labels)
ax.set_yticklabels(metric_names)

# Add text annotations
for i in range(len(metric_names)):
    for j in range(len(stages)):
        text = ax.text(j, i, f'{metrics_matrix[j][i]:.1f}%',
                      ha="center", va="center", color="black", fontweight='bold', fontsize=12)

ax.set_title("Reconstruction Accuracy Heatmap", fontsize=16, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Recall Rate (%)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "reconstruction_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR}/reconstruction_accuracy_heatmap.png")
plt.close()

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print(f"Output directory: {FIGURES_DIR}")
print("\nGenerated figures:")
print("  1. reconstruction_accuracy_overview.png - Comprehensive accuracy metrics")
print("  2. domain_coverage_analysis.png - Domain distribution per stage")
print("  3. stage4_sensitivity_analysis.png - Stage 4 detailed analysis")
print("  4. reconstruction_accuracy_heatmap.png - Overall performance heatmap")
print("="*80)
