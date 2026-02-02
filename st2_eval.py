import argparse
import json
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

SYSTEM_PROMPT = """
You are evaluating SemEval ST2 predictions. 
Decide whether PREDICTION refers to the same real-world entity as GOLD.

Rules:
1. Be tolerant of casing, abbreviations, minor spelling differences, synonyms, and singular/plural forms.
2. For products, ignore brand names or packaging if the underlying item is the same type (e.g., "Olive-Pomace Oil" ≈ "olive oil", "HAM, COOKED, SLICED" ≈ "ham slices").
3. For hazards, consider close variants or scientific/common synonyms equivalent (e.g., "E. COLI O157:H7" ≈ "escherichia coli").

Output ONLY valid JSON with a single key "correct" whose value is true or false:

{"correct": true|false}

""".strip()


def llm_judge(gold, pred) -> bool:
    prompt = json.dumps({
        "gold": "" if pd.isna(gold) else str(gold),
        "prediction": "" if pd.isna(pred) else str(pred)
    })

    payload = {
        "model": MODEL,
        "prompt": f"INPUT:\n{prompt}\nOUTPUT JSON:",
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()

    return json.loads(r.json()["response"])["correct"]


def evaluate(name, results):
    y_true = [True] * len(results)
    y_pred = results

    print(f"\n=== {name.upper()} METRICS (LLM-AS-JUDGE) ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    for col in ["hazard", "product"]:
        df[f"{col}_llm_correct"] = df.apply(
            lambda r: llm_judge(r[f"{col}_true"], r[f"{col}_pred"]),
            axis=1
        )

        evaluate(col, df[f"{col}_llm_correct"])

    out_path = args.csv_path.replace(".csv", "_llm_judged.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
