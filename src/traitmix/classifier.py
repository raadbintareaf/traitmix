"""Expressed-trait classifier: RoBERTa-base regressor fine-tuned on BIG5-CHAT trait labels,
validated on Essays-Big5. Falls back to lexical scoring when unavailable."""
from pathlib import Path
from .utils import ROOT

OUT = ROOT / "checkpoints" / "trait_classifier"

def train(max_rows=40000, epochs=1):
    try:
        import numpy as np, torch
        from datasets import load_dataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer)
    except ImportError as e:
        raise RuntimeError("Install transformers/datasets/torch to train the classifier.") from e
    from .personality import TRAITS
    ds = load_dataset("wenkai-li/big5_chat", split="train").shuffle(seed=0)
    ds = ds.select(range(min(max_rows, len(ds))))
    def lab(ex):
        y = [0.5] * 5
        t = str(ex.get("trait", "")).lower(); l = str(ex.get("level", "")).lower()
        if t in TRAITS:
            y[TRAITS.index(t)] = 1.0 if l == "high" else 0.0
        return {"text": ex.get("response") or ex.get("output") or "", "labels": y}
    ds = ds.map(lab, remove_columns=ds.column_names)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    ds = ds.map(lambda e: tok(e["text"], truncation=True, max_length=192), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=5, problem_type="regression")
    tr = Trainer(model=model, args=TrainingArguments(str(OUT), num_train_epochs=epochs,
                 per_device_train_batch_size=32, save_steps=1000, save_total_limit=2, report_to=[]),
                 train_dataset=ds, tokenizer=tok)
    ckpts = sorted(OUT.glob("checkpoint-*"))
    tr.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    tr.save_model(str(OUT / "final")); print("classifier saved:", OUT / "final")

def make_scorer():
    """Returns callable(texts)->{trait:score}; classifier if trained, else lexical fallback."""
    from .personality import lexical_trait_scores, TRAITS
    final = OUT / "final"
    if not final.exists():
        return lexical_trait_scores
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(final)); mdl = AutoModelForSequenceClassification.from_pretrained(str(final)).eval()
    def score(texts):
        if not texts:
            return {t: 0.5 for t in TRAITS}
        with torch.no_grad():
            enc = tok(texts[:64], truncation=True, max_length=192, padding=True, return_tensors="pt")
            out = torch.sigmoid(mdl(**enc).logits).mean(0).tolist()
        return dict(zip(TRAITS, [float(v) for v in out]))
    return score
