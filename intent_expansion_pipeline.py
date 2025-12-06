#!/usr/bin/env python3
"""
Intent Expansion Pipeline
=========================
Analyzes customer messages to discover missing or split-worthy intents.
Uses clustering + optional LLM refinement (Gemini 2.0 Flash Lite).

Usage:
    python intent_expansion_pipeline.py --input inputs_for_assignment.json --out_dir outputs
    python intent_expansion_pipeline.py --input inputs_for_assignment.json --out_dir outputs --use_llm
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Gemini LLM
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# =============================================================================
# Data Loading
# =============================================================================

def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_intent_mapper(data):
    """Extract intent hierarchy from data['intent_mapper']."""
    mapper = data.get("intent_mapper", [])
    intents = []
    for primary in mapper:
        p_id = primary.get("primary_intent_id", "")
        p_name = primary.get("primary_intent_name", "")
        for sec in primary.get("secondary_intents", []):
            intents.append({
                "primary_id": p_id,
                "primary_name": p_name,
                "secondary_id": sec.get("id", ""),
                "secondary_name": sec.get("name", ""),
                "description": sec.get("description", ""),
            })
    return intents


def extract_messages(data):
    """Extract customer messages from data['customer_messages']."""
    raw = data.get("customer_messages") or data.get("messages") or []
    if isinstance(data, list):
        raw = data

    messages = []
    for i, m in enumerate(raw):
        if isinstance(m, str):
            messages.append({"id": i, "text": m, "history": ""})
        else:
            text = m.get("current_human_message") or m.get("text") or m.get("message") or ""
            history = m.get("history", "")
            messages.append({"id": i, "text": text, "history": history})
    return messages


def simple_clean(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# Clustering
# =============================================================================

def cluster_messages(texts, min_k=5, max_k=15):
    """Cluster texts using TF-IDF + KMeans."""
    if len(texts) < min_k:
        return None

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(texts)

    best = {"k": None, "labels": None, "score": -1, "model": None}
    max_k_eff = min(max_k, len(texts) - 1)

    for k in range(min_k, max_k_eff + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best["score"]:
                best.update({"k": k, "labels": labels, "score": score, "model": km})
        except Exception:
            continue

    if best["k"] is None:
        return None
    return best["k"], best["labels"], best["score"], vectorizer, X, best["model"]


def get_cluster_info(X, labels, vectorizer, texts, top_n=8):
    """Get top terms and example texts per cluster."""
    terms = np.array(vectorizer.get_feature_names_out())
    clusters = {}

    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        sub = X[idx].mean(axis=0)
        subarr = np.asarray(sub).ravel()
        top_idx = subarr.argsort()[::-1][:top_n]
        top_terms = [terms[i] for i in top_idx]
        examples = [texts[i] for i in idx[:5]]

        clusters[c] = {
            "size": len(idx),
            "top_terms": top_terms,
            "examples": examples,
            "indices": idx.tolist(),
        }

    return clusters


# =============================================================================
# Intent Matching
# =============================================================================

def build_intent_vectors(intents, vectorizer):
    """Build TF-IDF vectors for intent descriptions."""
    texts = []
    for intent in intents:
        combined = f"{intent['secondary_name']} {intent['description']}"
        texts.append(simple_clean(combined))
    return vectorizer.transform(texts)


def match_cluster_to_intents(cluster_vec, intent_vecs, intents, threshold=0.15):
    """Find best matching intent for a cluster."""
    if intent_vecs.shape[0] == 0:
        return None, 0.0

    cluster_arr = np.asarray(cluster_vec)
    intent_arr = np.asarray(intent_vecs.todense()) if hasattr(intent_vecs, 'todense') else np.asarray(intent_vecs)

    sims = cosine_similarity(cluster_arr, intent_arr).flatten()
    best_idx = sims.argmax()
    best_sim = sims[best_idx]

    if best_sim >= threshold:
        return intents[best_idx], best_sim
    return None, best_sim


def make_suggested_name(terms):
    if not terms:
        return "candidate_intent"
    stopwords = {"the", "to", "for", "and", "or", "is", "it", "in", "on", "of", "a", "an", "be", "can", "you", "your", "i", "me", "my"}
    filtered = [t for t in terms if t.lower() not in stopwords and len(t) > 2][:3]
    if not filtered:
        filtered = terms[:2]
    slug = "_".join(re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in filtered)
    return slug[:50] or "candidate_intent"


# =============================================================================
# LLM Integration (Gemini)
# =============================================================================

class GeminiNamer:
    """Uses Gemini to generate better intent names and descriptions."""

    def __init__(self, api_key, model_name="gemini-2.0-flash-lite"):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.context = ""

    def set_context(self, intents):
        self.context = "\n".join([
            f"- {i['primary_name']} > {i['secondary_name']}: {i['description']}"
            for i in intents[:20]
        ])

    def generate(self, top_terms, examples, recommendation, matched_intent):
        prompt = f"""You are an expert at designing intent taxonomies for conversational AI.

EXISTING INTENTS:
{self.context}

TASK: Analyze this cluster and suggest an intent name.

TOP KEYWORDS: {', '.join(top_terms)}

EXAMPLE MESSAGES:
{chr(10).join(f'- "{m}"' for m in examples[:5])}

STATUS: {recommendation}
{f"PARTIALLY MATCHES: {matched_intent}" if matched_intent else "NO EXISTING INTENT MATCH"}

RULES:
1. Intent name: 2-4 words, snake_case (e.g., product_usage, order_tracking)
2. Distinct from existing intents
3. Description: 1 sentence explaining when this intent applies

OUTPUT FORMAT (JSON only, no markdown):
{{"intent_name": "your_intent_name", "description": "One sentence description."}}
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=150)
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```(?:json)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            result = json.loads(text)
            return result.get("intent_name", "unknown"), result.get("description", "")
        except Exception as e:
            print(f"    [LLM Warning] {e}")
            return make_suggested_name(top_terms), f"Messages about {', '.join(top_terms[:3])}"


# =============================================================================
# Suggestion Generation
# =============================================================================

def generate_suggestions(df, intents, min_cluster_size=8, silhouette_threshold=0.1, use_llm=False):
    """Main suggestion generation logic."""
    texts = df["text_clean"].tolist()
    original_texts = df["text"].tolist()

    result = cluster_messages(texts, min_k=5, max_k=15)
    if result is None:
        return [], {}

    k, labels, score, vectorizer, X, model = result
    cluster_info = get_cluster_info(X, labels, vectorizer, original_texts)

    intent_vecs = build_intent_vectors(intents, vectorizer) if intents else None

    suggestions = []
    stats = {
        "total_messages": len(texts),
        "num_clusters": k,
        "silhouette_score": round(score, 3),
        "llm_enabled": use_llm,
    }

    # Initialize LLM if enabled
    llm_namer = None
    if use_llm:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key and GENAI_AVAILABLE:
            try:
                llm_namer = GeminiNamer(api_key)
                llm_namer.set_context(intents)
                print("  LLM naming enabled (Gemini 2.0 Flash Lite)")
            except Exception as e:
                print(f"  [Warning] Could not initialize LLM: {e}")
        else:
            if not api_key:
                print("  [Warning] GEMINI_API_KEY not found in environment")
            if not GENAI_AVAILABLE:
                print("  [Warning] google-generativeai not installed")

    for cluster_id, info in cluster_info.items():
        if info["size"] < min_cluster_size:
            continue

        cluster_indices = info["indices"]
        cluster_vec = X[cluster_indices].mean(axis=0)

        matched_intent, similarity = None, 0.0
        if intent_vecs is not None and len(intents) > 0:
            matched_intent, similarity = match_cluster_to_intents(cluster_vec, intent_vecs, intents)

        # Determine recommendation
        if matched_intent and similarity >= 0.3:
            recommendation = "MAPS_TO_EXISTING"
            justification = f"Cluster maps well to existing intent '{matched_intent['secondary_name']}' (similarity={similarity:.2f})."
        elif matched_intent and similarity >= 0.15:
            recommendation = "CONSIDER_SPLIT"
            justification = f"Cluster partially overlaps with '{matched_intent['secondary_name']}' (similarity={similarity:.2f}). Consider creating a sub-intent."
        else:
            recommendation = "NEW_INTENT_CANDIDATE"
            justification = f"Cluster does not match any existing intent well (best similarity={similarity:.2f}). Potential new intent covering {info['size']} messages."

        # Generate name (with LLM if available and not MAPS_TO_EXISTING)
        if llm_namer and recommendation != "MAPS_TO_EXISTING":
            print(f"    Generating LLM name for cluster {cluster_id}...")
            intent_name, intent_desc = llm_namer.generate(
                info["top_terms"], info["examples"], recommendation,
                matched_intent['secondary_name'] if matched_intent else None
            )
            time.sleep(0.5)  # Rate limiting
        else:
            intent_name = make_suggested_name(info["top_terms"])
            intent_desc = ""

        suggestion = {
            "cluster_id": int(cluster_id),
            "cluster_size": info["size"],
            "cluster_fraction": round(info["size"] / len(texts), 3),
            "top_terms": info["top_terms"],
            "example_messages": info["examples"],
            "suggested_intent_name": intent_name,
            "suggested_intent_description": intent_desc,
            "matched_existing_intent": {
                "primary": matched_intent["primary_name"],
                "secondary": matched_intent["secondary_name"],
                "description": matched_intent["description"],
            } if matched_intent else None,
            "match_similarity": round(similarity, 3),
            "recommendation": recommendation,
            "justification": justification,
        }
        suggestions.append(suggestion)

    # Sort by priority
    priority = {"NEW_INTENT_CANDIDATE": 0, "CONSIDER_SPLIT": 1, "MAPS_TO_EXISTING": 2}
    suggestions.sort(key=lambda x: (priority.get(x["recommendation"], 3), -x["cluster_size"]))

    return suggestions, stats


# =============================================================================
# Output Generation
# =============================================================================

def generate_markdown_report(suggestions, stats, intents, out_path):
    lines = []
    lines.append("# Intent Expansion Pipeline Report\n")

    lines.append("## Summary\n")
    lines.append(f"- **Total messages analyzed**: {stats.get('total_messages', 0)}")
    lines.append(f"- **Number of clusters**: {stats.get('num_clusters', 0)}")
    lines.append(f"- **Clustering quality (silhouette)**: {stats.get('silhouette_score', 0)}")
    lines.append(f"- **LLM Refinement**: {'Enabled ✓' if stats.get('llm_enabled') else 'Disabled'}")
    lines.append(f"- **Existing intents in mapper**: {len(intents)}")
    lines.append("")

    rec_counts = Counter(s["recommendation"] for s in suggestions)
    lines.append("### Recommendation Breakdown\n")
    lines.append(f"- 🆕 **NEW_INTENT_CANDIDATE**: {rec_counts.get('NEW_INTENT_CANDIDATE', 0)}")
    lines.append(f"- ✂️ **CONSIDER_SPLIT**: {rec_counts.get('CONSIDER_SPLIT', 0)}")
    lines.append(f"- ✅ **MAPS_TO_EXISTING**: {rec_counts.get('MAPS_TO_EXISTING', 0)}")
    lines.append("")

    lines.append("---\n")
    lines.append("## Detailed Suggestions\n")

    for i, s in enumerate(suggestions, 1):
        emoji = {"NEW_INTENT_CANDIDATE": "🆕", "CONSIDER_SPLIT": "✂️", "MAPS_TO_EXISTING": "✅"}.get(s["recommendation"], "❓")
        lines.append(f"### {i}. {emoji} {s['recommendation']}: `{s['suggested_intent_name']}`\n")

        if s.get("suggested_intent_description"):
            lines.append(f"**Description**: {s['suggested_intent_description']}\n")

        lines.append(f"- **Cluster size**: {s['cluster_size']} ({s['cluster_fraction']*100:.1f}% of messages)")
        lines.append(f"- **Top terms**: {', '.join(s['top_terms'][:6])}")

        if s["matched_existing_intent"]:
            mi = s["matched_existing_intent"]
            lines.append(f"- **Best match**: {mi['primary']} → {mi['secondary']} (similarity: {s['match_similarity']})")

        lines.append(f"\n**Justification**: {s['justification']}\n")

        lines.append("**Example messages**:")
        for ex in s["example_messages"][:3]:
            lines.append(f'  - "{ex[:100]}{"..." if len(ex) > 100 else ""}"')
        lines.append("")

    lines.append("---\n")
    lines.append("## Existing Intent Mapper Reference\n")
    by_primary = defaultdict(list)
    for intent in intents:
        by_primary[intent["primary_name"]].append(intent)

    for primary, secs in by_primary.items():
        lines.append(f"### {primary}\n")
        for sec in secs:
            desc = sec['description'][:80] + "..." if len(sec['description']) > 80 else sec['description']
            lines.append(f"- **{sec['secondary_name']}** (`{sec['secondary_id']}`): {desc}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Intent Expansion Pipeline")
    parser.add_argument("--input", default="inputs_for_assignment.json", help="Input JSON file")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--min_cluster_size", type=int, default=8, help="Minimum messages per cluster")
    parser.add_argument("--use_llm", action="store_true", help="Use Gemini LLM for intent naming")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("INTENT EXPANSION PIPELINE")
    print(f"{'='*60}\n")

    print(f"[1/4] Loading data from {args.input}...")
    data = load_input(args.input)
    intents = extract_intent_mapper(data)
    messages = extract_messages(data)
    print(f"  Found {len(intents)} existing intents")
    print(f"  Found {len(messages)} customer messages")

    print(f"\n[2/4] Preparing data...")
    df = pd.DataFrame(messages)
    df["text_clean"] = df["text"].fillna("").apply(simple_clean)

    print(f"\n[3/4] Clustering and generating suggestions...")
    if args.use_llm:
        print("  LLM mode enabled")
    suggestions, stats = generate_suggestions(df, intents, min_cluster_size=args.min_cluster_size, use_llm=args.use_llm)

    print(f"\n[4/4] Saving outputs...")
    json_out = os.path.join(args.out_dir, "intent_suggestions.json")
    md_out = os.path.join(args.out_dir, "intent_suggestions.md")

    output_data = {
        "stats": stats,
        "suggestions": suggestions,
        "existing_intents": intents,
    }
    save_json(output_data, json_out)
    generate_markdown_report(suggestions, stats, intents, md_out)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ JSON: {json_out}")
    print(f"  ✅ Report: {md_out}")

    rec_counts = Counter(s["recommendation"] for s in suggestions)
    print(f"\n📊 Summary:")
    print(f"   🆕 NEW_INTENT_CANDIDATE: {rec_counts.get('NEW_INTENT_CANDIDATE', 0)}")
    print(f"   ✂️  CONSIDER_SPLIT: {rec_counts.get('CONSIDER_SPLIT', 0)}")
    print(f"   ✅ MAPS_TO_EXISTING: {rec_counts.get('MAPS_TO_EXISTING', 0)}")

    if args.use_llm and stats.get("llm_enabled"):
        print(f"\n   🤖 LLM refinement was applied")


if __name__ == "__main__":
    main()
