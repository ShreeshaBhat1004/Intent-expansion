# Intent Expansion Pipeline

Files added:

- `intent_expansion_pipeline.py`: Python script that analyzes messages and proposes candidate intent splits.
- `requirements.txt`: Python deps.

Quick run:

```bash
python3 -m pip install -r requirements.txt
python3 intent_expansion_pipeline.py --input inputs_for_assignment.json --out_dir outputs
```

Outputs:

- `outputs/intent_suggestions.json` — machine-readable suggestions
- `outputs/intent_suggestions.md` — human-readable report

Notes:

- The script uses TF-IDF + KMeans clustering per (primary -> secondary) group to surface high-level split candidates.
- Optional LLM refinement hooks can be added to refine suggestion names and descriptions.
