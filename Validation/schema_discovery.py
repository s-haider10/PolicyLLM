"""Schema discovery: cluster actions across policies via Sentence-BERT embeddings."""
import json
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util


def schema_discovery(
    json_list: List[Dict[str, Any]],
    cosine_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Process policies to discover canonical action clusters.

    Extracts action strings, generates Sentence-BERT embeddings, clusters them,
    and adds ``canonical_actions`` to qualifying policies. Policies with
    ``discovery.human_validated == False`` are skipped.

    Args:
        json_list: List of policy dicts (v1.0 schema).
        cosine_threshold: Minimum cosine similarity for clustering.

    Returns:
        Modified policy list with ``canonical_actions`` added where applicable.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    processed = json.loads(json.dumps(json_list))

    # Filter: skip policies where discovery.human_validated is false
    eligible_indices: List[int] = []
    for i, obj in enumerate(processed):
        if not isinstance(obj, dict):
            continue
        discovery = obj.get("discovery")
        if isinstance(discovery, dict) and discovery.get("human_validated") is False:
            continue
        eligible_indices.append(i)

    # Extract action strings
    all_actions: List[str] = []
    action_to_obj_map: List[Dict[str, Any]] = []
    for i in eligible_indices:
        obj = processed[i]
        actions_field = obj.get("actions")
        if not isinstance(actions_field, list):
            continue
        for action_obj in actions_field:
            if isinstance(action_obj, dict):
                action_str = action_obj.get("action")
            elif isinstance(action_obj, str):
                action_str = action_obj
            else:
                action_str = None
            if action_str:
                all_actions.append(action_str)
                action_to_obj_map.append({"obj_index": i, "action": action_str})

    if not all_actions:
        return processed

    # Embed and cluster
    action_embeddings = model.encode(all_actions, convert_to_tensor=True)
    action_clustered = [False] * len(all_actions)
    clusters: List[Dict[str, Any]] = []

    for i, emb_i in enumerate(action_embeddings):
        if action_clustered[i]:
            continue
        current_cluster = [all_actions[i]]
        action_clustered[i] = True
        cluster_mappings = [action_to_obj_map[i]]

        for j, emb_j in enumerate(action_embeddings):
            if i == j or action_clustered[j]:
                continue
            score = util.cos_sim(emb_i, emb_j)
            if score > cosine_threshold:
                current_cluster.append(all_actions[j])
                action_clustered[j] = True
                cluster_mappings.append(action_to_obj_map[j])

        clusters.append({"actions": current_cluster, "mappings": cluster_mappings})

    # Keep only cross-policy clusters (span >= 2 distinct policies)
    cross_policy_clusters = [
        c for c in clusters
        if len({m["obj_index"] for m in c["mappings"]}) >= 2
    ]

    # Assign canonical actions
    for cluster in cross_policy_clusters:
        canonical_var = cluster["actions"][0]
        for mapping in cluster["mappings"]:
            policy = processed[mapping["obj_index"]]
            if "canonical_actions" not in policy:
                policy["canonical_actions"] = []
            if canonical_var not in policy["canonical_actions"]:
                policy["canonical_actions"].append(canonical_var)

    return processed
