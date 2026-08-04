"""T-arm: QLoRA fine-tuning of the base model on BIG5-CHAT with trait control tags.
Resumable (HF Trainer checkpoints). Degrades gracefully if the GPU stack is absent."""
from pathlib import Path
from .utils import ROOT
from . import personality as pers

OUT = ROOT / "checkpoints" / "qlora_big5chat"

def build_dataset(max_rows=None):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install the GPU stack (see requirements.txt comments): pip install datasets transformers peft trl accelerate bitsandbytes") from e
    ds = load_dataset("wenkai-li/big5_chat", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    lvl = {"high": "very_high", "low": "very_low"}
    def to_text(ex):
        trait = str(ex.get("trait", "openness")).lower()
        level = lvl.get(str(ex.get("level", "high")).lower(), "average")
        tag = f"<{pers.SHORT.get(trait,'O')}={level}>"
        instr = ex.get("instruction") or ex.get("scenario") or ""
        resp = ex.get("response") or ex.get("output") or ""
        return {"text": f"[personality control] {tag}\nUser: {instr}\nAssistant: {resp}"}
    return ds.map(to_text, remove_columns=[c for c in ds.column_names])

def train(base_model="meta-llama/Llama-3.1-8B-Instruct", max_rows=None, epochs=1,
          per_device_bs=2, grad_accum=8, resume=True):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig
        from trl import SFTTrainer
    except ImportError as e:
        raise RuntimeError("GPU stack missing; install per requirements.txt comments.") from e
    ds = build_dataset(max_rows)
    tok = AutoTokenizer.from_pretrained(base_model); tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto")
    args = TrainingArguments(output_dir=str(OUT), num_train_epochs=epochs,
                             per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=grad_accum,
                             learning_rate=1e-4, logging_steps=25, save_steps=500, save_total_limit=3,
                             bf16=True, report_to=[])
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, dataset_text_field="text",
                         peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
                         max_seq_length=512, tokenizer=tok)
    ckpts = sorted(OUT.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if (resume and ckpts) else None)
    trainer.save_model(str(OUT / "final"))
    print("Adapter saved to", OUT / "final",
          "\nServe with: vllm serve", base_model, "--enable-lora --lora-modules big5=" + str(OUT / "final"))
