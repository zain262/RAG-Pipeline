import argparse
import json
import re
import traceback
from typing import List, Any, Optional

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

# Optional metrics
try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None



def clean_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)  
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_input_text(title: Any, text: Any) -> str:
    t = clean_text(title)
    x = clean_text(text)
    if t and x:
        return f"{t}\n\n{x}"
    return t or x


def extract_json(text: str) -> str:
    """Pull a JSON object out of model output (handles extra chatter/fences)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0).strip() if m else t


def norm_label(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s or None


def map_to_allowed(pred_norm: Optional[str], allowed_raw: List[str]) -> Optional[str]:
    """
    Return the closest allowed label (raw form) or None.
    - exact match on normalized labels
    - containment fallback
    """
    if pred_norm is None:
        return None

    allowed_norm = [norm_label(a) for a in allowed_raw]

    # exact
    if pred_norm in allowed_norm:
        return allowed_raw[allowed_norm.index(pred_norm)]

    # containment fallback
    for i, a in enumerate(allowed_norm):
        if a and (pred_norm in a or a in pred_norm):
            return allowed_raw[i]

    return None


def chunk_text(s: str, chunk_size: int = 2000, overlap: int = 250) -> List[str]:
    """Simple character-based chunker for retrieval."""
    s = (s or "").strip()
    if not s:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    out = []
    i = 0
    while i < len(s):
        out.append(s[i : i + chunk_size])
        i += (chunk_size - overlap)
    return out


def parse_json_strict(text: str) -> dict:
    """Parse model output into a dict. Raises on failure."""
    j = extract_json(text)
    obj = json.loads(j)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON is not an object")
    return obj


@component
class PromptToMessages:
    @component.output_types(messages=List[ChatMessage])
    def run(self, prompt: str):
        return {"messages": [ChatMessage.from_user(prompt)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)

    # LLM (Ollama)
    ap.add_argument("--model", type=str, default="llama3.1:8b")
    ap.add_argument("--api_base_url", type=str, default="http://localhost:11434/v1")

    # Retrieval
    ap.add_argument("--top_k", type=int, default=8)  # increased (was 1)
    ap.add_argument("--max_docs", type=int, default=3000)
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--chunk_overlap", type=int, default=250)

    # Run control
    ap.add_argument("--run_splits", type=str, default="validation,test")
    ap.add_argument("--out_csv", type=str, default="predictions_st1.csv")
    ap.add_argument("--truncate_report_chars", type=int, default=1200)

    # Generation
    ap.add_argument("--max_tokens", type=int, default=60)  # increased a bit (was 30)
    args = ap.parse_args()

    df = pd.read_csv(args.data_path)

    needed = ["title", "text", "hazard-category", "product-category", "semeval-split"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df["input_text"] = df.apply(lambda r: build_input_text(r["title"], r["text"]), axis=1)

    train_df = df[df["semeval-split"].astype(str).str.lower().eq("training")].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows found (semeval-split == training).")

    hazard_labels = sorted(train_df["hazard-category"].dropna().unique().tolist())
    product_labels = sorted(train_df["product-category"].dropna().unique().tolist())


    store = InMemoryDocumentStore()
    kb_docs: List[Document] = []

    for i, row in train_df.head(args.max_docs).iterrows():
        txt = row.get("input_text")
        if not isinstance(txt, str) or len(txt) < 50:
            continue

        kb_docs.append(
            Document(
                content=txt,
                id=f"train_{i}",
                meta={"train_row_id": int(i)},
            )
        )

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="BAAI/bge-large-en-v1.5"
    )
    doc_embedder.warm_up()
    kb_docs = doc_embedder.run(kb_docs)["documents"]
    store.write_documents(kb_docs, policy="overwrite")

    text_embedder = SentenceTransformersTextEmbedder(
        model="BAAI/bge-large-en-v1.5"
    )
    retriever = InMemoryEmbeddingRetriever(document_store=store)

    prompt = r"""
You are doing SemEval Task 9 ST1 (coarse classification).
Pick EXACTLY ONE label from each allowed list.

HARD RULES:
- Output ONLY valid JSON. No markdown. No extra text.
- Keys must be exactly: "hazard_category", "product_category"
- Each value MUST be copied EXACTLY from the allowed lists below (case-sensitive).
- Do NOT output anything else.

Allowed hazard_category:
{{ hazard_labels }}

Allowed product_category:
{{ product_labels }}

Recall:
{{ report_text }}

Evidence (similar recalls):
{% for d in documents %}
- {{ d.content[:250] }}
{% endfor %}

JSON:
"""

    prompt_builder = PromptBuilder(
        template=prompt,
        required_variables=["hazard_labels", "product_labels", "report_text", "documents"],
    )

    prompt_to_messages = PromptToMessages()

    generator = OpenAIChatGenerator(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key=Secret.from_token("ollama"),
        generation_kwargs={
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        },
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("prompt_to_messages", prompt_to_messages)
    pipe.add_component("generator", generator)

    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "prompt_to_messages.prompt")
    pipe.connect("prompt_to_messages.messages", "generator.messages")


    splits = [s.strip().lower() for s in args.run_splits.split(",") if s.strip()]
    run_df = df[df["semeval-split"].astype(str).str.lower().isin(splits)].copy()

    rows = []
    bad_json = 0
    bad_label = 0

    for idx, row in run_df.iterrows():
        report_text = row["input_text"]
        if args.truncate_report_chars and isinstance(report_text, str):
            report_text = report_text[: args.truncate_report_chars]

        try:
            result = pipe.run(
                {
                    "text_embedder": {"text": report_text},
                    "retriever": {"top_k": args.top_k},
                    "prompt_builder": {
                        "hazard_labels": hazard_labels,
                        "product_labels": product_labels,
                        "report_text": report_text,
                    },
                }
            )

            raw = result["generator"]["replies"][0].text

            # Parse JSON; retry once if invalid
            pred_h_raw = None
            pred_p_raw = None
            parsed = None

            try:
                parsed = parse_json_strict(raw)
            except Exception:
                bad_json += 1
                retry_prompt = (
                    "Your previous output was invalid JSON or contained invalid labels.\n"
                    "Output ONLY valid JSON with keys hazard_category and product_category.\n"
                    "Each value MUST be copied EXACTLY from the allowed lists.\n\n"
                    f"Allowed hazard_category:\n{hazard_labels}\n\n"
                    f"Allowed product_category:\n{product_labels}\n\n"
                    f"Recall:\n{report_text}\n\n"
                    "JSON:"
                )
                retry_res = generator.run(messages=[ChatMessage.from_user(retry_prompt)])
                raw = retry_res["replies"][0].text
                parsed = parse_json_strict(raw)

            pred_h_raw = parsed.get("hazard_category")
            pred_p_raw = parsed.get("product_category")

            pred_h = map_to_allowed(norm_label(pred_h_raw), hazard_labels)
            pred_p = map_to_allowed(norm_label(pred_p_raw), product_labels)

            if pred_h not in hazard_labels or pred_p not in product_labels:
                bad_label += 1

            rows.append(
                {
                    "row_id": idx,
                    "semeval-split": row["semeval-split"],
                    "hazard-category_true": row.get("hazard-category", None),
                    "product-category_true": row.get("product-category", None),
                    "hazard-category_pred": pred_h,
                    "product-category_pred": pred_p,
                    "raw_hazard_pred": pred_h_raw,
                    "raw_product_pred": pred_p_raw,
                    "raw_model_output": raw,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "row_id": idx,
                    "semeval-split": row["semeval-split"],
                    "hazard-category_true": row.get("hazard-category", None),
                    "product-category_true": row.get("product-category", None),
                    "hazard-category_pred": None,
                    "product-category_pred": None,
                    "raw_model_output": None,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)


    if f1_score is not None:
        val = out[out["semeval-split"].astype(str).str.lower().eq("validation")].dropna(
            subset=[
                "hazard-category_true",
                "hazard-category_pred",
                "product-category_true",
                "product-category_pred",
            ]
        )
        if len(val) > 0:
            h_f1 = f1_score(val["hazard-category_true"], val["hazard-category_pred"], average="macro")
            p_f1 = f1_score(val["product-category_true"], val["product-category_pred"], average="macro")
            print(f"validation macro-F1 hazard={h_f1:.4f} product={p_f1:.4f}")


if __name__ == "__main__":
    main()
