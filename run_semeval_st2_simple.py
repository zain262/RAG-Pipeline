import argparse
import json
import re
import subprocess
import sys
import traceback
from typing import Any, Optional, List

import pandas as pd

from haystack import Document, Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret



from sklearn.metrics import accuracy_score



def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_input_text(
    title: Any,
    text: Any,
    year: Any = None,
    month: Any = None,
    day: Any = None,
    country: Any = None,
) -> str:
    parts = []
    if year:
        parts.append(f"Year: {year}")
    if month:
        parts.append(f"Month: {month}")
    if day:
        parts.append(f"Day: {day}")
    if country:
        parts.append(f"Country: {country}")
    if title:
        parts.append(normalize_text(title))
    if text:
        parts.append(normalize_text(text))
    return "\n\n".join(parts).strip()


def extract_json(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0).strip() if m else t.strip()


def find_span_ci(hay: str, needle: Optional[str]) -> Optional[str]:
    if needle is None:
        return None
    needle = str(needle).strip()
    if not needle or needle.lower() == "null":
        return None

    m = re.search(re.escape(needle), hay, flags=re.IGNORECASE)
    if m:
        return hay[m.start() : m.end()]

    words = needle.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            subphrase = " ".join(words[start : start + length])
            m = re.search(re.escape(subphrase), hay, flags=re.IGNORECASE)
            if m:
                return hay[m.start() : m.end()]

    return None


def parse_json_strict(text: str) -> dict:
    j = extract_json(text)
    obj = json.loads(j)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object")
    return obj


def norm_for_f1(x: Any) -> str:

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "__NULL__"
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s if s else "__NULL__"


@component
class PromptToChatMessages:
    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str):
        return {"messages": [ChatMessage.from_user(prompt)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--run_splits", default="validation")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--api_base_url", default="http://localhost:11434/v1")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_docs", type=int, default=3000)
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--out_csv", default="predictions_st2.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data_path)

    for c in ["title", "text", "semeval-split"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    has_gold_hazard = "hazard" in df.columns
    has_gold_product = "product" in df.columns

    df["input_text"] = df.apply(
        lambda r: build_input_text(
            r.get("title"),
            r.get("text"),
            year=r.get("YEAR"),
            month=r.get("MONTH"),
            day=r.get("DAY"),
            country=r.get("COUNTRY"),
        ),
        axis=1,
    )

    splits = [s.strip().lower() for s in args.run_splits.split(",") if s.strip()]
    run_df = df[df["semeval-split"].astype(str).str.lower().isin(splits)].copy()
    if args.limit_rows and args.limit_rows > 0:
        run_df = run_df.head(args.limit_rows)

    train_df = df[df["semeval-split"].astype(str).str.lower() == "training"].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows found (semeval-split == training).")


    store = InMemoryDocumentStore()
    kb_docs: List[Document] = []

    for i, row in train_df.head(args.max_docs).iterrows():
        txt = row.get("input_text")
        if not isinstance(txt, str) or len(txt) < 50:
            continue
        kb_docs.append(Document(content=txt, id=f"train_{i}", meta={"train_row_id": int(i)}))


    # Embed your documents with the new model
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    kb_docs = doc_embedder.run(kb_docs)["documents"]
    store.write_documents(kb_docs, policy="overwrite")

    # Use the same model for query embeddings
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store=store)


    prompt_template = r"""
You are doing SemEval Task 9 ST2 (mention extraction). TASK: From the recall text below, extract:
- hazard: the most specific hazard mention explicitly stated in the text
- product: the food product mention explicitly stated in the text

HARD RULES:
- Output ONLY valid JSON. No markdown. No extra text.
- Keys must be exactly: "hazard", "product"
- Each value MUST be an exact substring copied from the recall text (copy/paste)
- If a value is not an exact substring, output null
- Do NOT paraphrase or expand into a sentence
- Avoid generic "contamination" if a specific hazard exists
- Do NOT output only a company/brand name as the product if an actual food product is mentioned
- If you cannot find an exact substring for a field, use null

Recall text:
{{ report_text }}

Evidence (similar past recalls):
{% for d in documents %}
- {{ d.content[:220] }}
{% endfor %}

Return JSON:
"""

    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["report_text", "documents"],
    )

    generator = OpenAIChatGenerator(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key=Secret.from_token("ollama"),
        generation_kwargs={"max_tokens": args.max_tokens, "temperature": 0.0},
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    #pipe.add_component("reranker", reranker)  
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("prompt_to_messages", PromptToChatMessages())
    pipe.add_component("generator", generator)

    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "prompt_to_messages.prompt")
    pipe.connect("prompt_to_messages.messages", "generator.messages")


    out_rows = []
    bad_json = 0

    for idx, row in run_df.iterrows():
        report_text = row["input_text"]
        try:
            res = pipe.run(
                {
                    "text_embedder": {"text": report_text},
                    "retriever": {"top_k": args.top_k},
                    "prompt_builder": {"report_text": report_text},
                }
            )
            raw = res["generator"]["replies"][0].text

            try:
                parsed = parse_json_strict(raw)
            except Exception:
                bad_json += 1
                retry_prompt = (
                    "Your previous output was invalid JSON.\n"
                    "Output ONLY valid JSON (no markdown, no extra text).\n"
                    'Keys must be exactly "hazard" and "product".\n'
                    "Each value MUST be an exact substring copied from the Recall text or null.\n\n"
                    f"Recall text:\n{report_text}\n\n"
                    "Return JSON:"
                )
                retry_direct = generator.run(messages=[ChatMessage.from_user(retry_prompt)])
                raw = retry_direct["replies"][0].text
                parsed = parse_json_strict(raw)

            hz_raw = parsed.get("hazard", None)
            pr_raw = parsed.get("product", None)

            hz_txt = find_span_ci(report_text, hz_raw)
            pr_txt = find_span_ci(report_text, pr_raw)

            out_rows.append(
                {
                    "row_id": idx,
                    "semeval-split": row.get("semeval-split"),
                    "hazard_pred": hz_txt,
                    "product_pred": pr_txt,
                    "raw_model_output": parsed,
                    "hazard_true": row.get("hazard") if has_gold_hazard else None,
                    "product_true": row.get("product") if has_gold_product else None,
                }
            )

        except Exception:
            out_rows.append(
                {
                    "row_id": idx,
                    "semeval-split": row.get("semeval-split"),
                    "hazard_pred": None,
                    "product_pred": None,
                    "raw_model_output": None,
                    "hazard_true": row.get("hazard") if has_gold_hazard else None,
                    "product_true": row.get("product") if has_gold_product else None,
                    "error": traceback.format_exc(),
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")
    print("\nRunning ST2 LLM evaluation...")
    subprocess.run([sys.executable, "st2_eval.py", args.out_csv], check=True)



if __name__ == "__main__":
    main()
