# Approach and Design

Overview
- Goal: build an extensible pipeline that analyzes existing message data and suggests where existing intent categories should be split or refined.
- Deliverables: `intent_expansion_pipeline.py`, report outputs in `outputs/`, and this document describing approach, guardrails, and limitations.

Workflow architecture
- Input: JSON containing `intent_mapper` and `messages`. The script is defensive about input shapes.
- Preprocessing: light normalization (lowercase, strip punctuation / URLs) to keep token counts low for small LLMs.
- Representation: sparse TF-IDF with 1-2 grams (efficient for thousands of short messages).
- Per-intent analysis: group by `primary -> secondary` pair and run clustering only where group size >= threshold (default 20) to avoid noisy splits.
- Clustering: KMeans for k in [2..5], pick k with best silhouette score; report clusters with size >= 8% of group.
- Candidate generation: for each cluster, extract representative terms (top TF-IDF) and form a short suggested secondary-intent slug.

Why this design
- Scalability: TF-IDF + per-intent clustering keeps memory usage and LLM calls small; can process thousands of messages by streaming groups.
- Determinism: core pipeline is deterministic and reproducible; LLM is optional and only used for human-friendly names/definitions.
- Precision-first: only propose splits when there is statistical evidence (silhouette score, cluster fraction).

Guardrails & fallbacks
- Minimum examples per group: avoid splitting categories with < 20 examples.
- Minimum cluster fraction: ignore tiny clusters < 8% to avoid proliferation of micro-intents.
- Silhouette check: classify as `good_silhouette` or `low_silhouette` to indicate confidence.
- Ambiguity: if clusters are weak, pipeline still reports suggestions but marks them as low confidence — do not auto-deploy such splits.
- Default behavior: For ambiguous or low-volume cases, keep the original intent; new intents must be manually reviewed before adding to production.

LLM integration (optional)
- The pipeline provides hooks to call an LLM to refine the suggested intent name and produce a short description and examples per candidate.
- Keep the prompt small: pass representative 3–5 example messages and the top TF-IDF terms to the LLM.
- Limit tokens and calls: refactor candidate-level calls so a single call handles batched candidates to control cost.

Failure cases & limitations
- Short messages with few tokens (e.g., "Where is my order?") may cluster poorly with TF-IDF. Consider embeddings (sentence-transformers) as an improvement.
- Highly overlapping concerns: customers asking about both refunds and cancellations in same message will be noisy; adding a multi-label or slot extraction stage helps.
- Small intents: long tail intents will be missed unless data grows; consider an active sampling process to surface rare intents.

Extensions and next steps
- Replace TF-IDF with embeddings (sentence-transformers) for semantic clustering (better generalization at slightly higher cost).
- Add automatic evaluation: sample suggested splits and ask crowd or use heuristic labels to measure if split improves downstream intent-classifier accuracy.
- Add rules for naming and for mapping new secondary intents to analytics dashboards and routing logic.

How to review suggestions
1. Run the pipeline locally to produce `outputs/intent_suggestions.md`.
2. Inspect high-confidence suggestions (`silhouette >= 0.25`, cluster_fraction >= ~10%).
3. For each candidate: create acceptance criteria, example utterances, and sample training data before adding to intent mapper.
