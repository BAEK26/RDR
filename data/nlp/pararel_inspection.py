#!/usr/bin/env python
# run_llama31_pararel_batched.py
"""
Infer PARAREL object predictions with Meta-Llama-3.1-Instruct — batched,
with per-sample prompt-length slicing.
"""

import json
import argparse
import re
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
debug = False
def build_chat_messages(ex):
    subject: str = ex["subject"]
    template: str = ex["template"]
    candidates = ex["candidates"]
    user_sentence = (
        "answer candidates:" + ", ".join(candidates) + "\n" +
        ex["query"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable extraction assistant. Given a list of answer candidates and a user query, pick exactly one entity from the candidates that best completes the query. Respond with only that entity, without any extra commentary."
            ),
        },
        {"role": "user", "content": user_sentence},
    ]

def postprocess(text: str) -> str:
    if "assistant" in text:
        # Llama-3.1-Instruct sometimes adds "assistant" at the start
        text = text.split("assistant", 1)[-1]
    text = text.strip().split("\n")[0]
    return re.sub(r'^[\"\'“”‘’\s]+|[\"\'“”‘’\s]+$', "", text)

def collate_fn(examples):
    return examples  # we’ll batch‐tokenize manually

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--outfile",
                        default="pararel_ent_llama31_predictions.jsonl")
    args = parser.parse_args()

    print("Loading tokenizer & model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    gen_cfg = GenerationConfig(max_new_tokens=16)
    model.config.pad_token_id = tokenizer.eos_token_id
    print("Loading PARAREL patterns …")
    ds = load_dataset("coastalcph/pararel_patterns", split="train")
    # ds = ds.filter(
    #     lambda ex: 'edison' in ex["subject"].lower(),
    #     # exclude the trivial template
    # )
    if args.max_rows:
        ds = ds.select(range(args.max_rows))

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)

    results = []
    correct = 0
    debug_whole = []
    debug_index = []

    for batch in tqdm(loader, desc="infer"):
        # 1) build raw prompts
        messages = [build_chat_messages(ex)
                    for ex in batch]
        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) compute each prompt’s true length (no padding)
        raw_enc = tokenizer(prompts, padding=False, truncation=True)
        prompt_lengths = [len(ids) for ids in raw_enc["input_ids"]]

        # 3) tokenize with padding for batch
        inputs = tokenizer(prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=False
                           ).to(model.device)

        # 4) generate continuations
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            generation_config=gen_cfg,
        )
        # outputs shape: [bs, padded_prompt_len + gen_len]

        # 5) slice per-sample based on its prompt_length
        preds = []
        for i, length in enumerate(prompt_lengths):
            gen_ids = outputs[i, length:]              # only the new tokens
            text = tokenizer.decode(gen_ids,
                                    skip_special_tokens=True)
            debug_text = tokenizer.decode(outputs[i],
                                          skip_special_tokens=True)
            if debug:
                debug_whole.append(debug_text)
                debug_index.append(length)
                if i == 4:
                    import pdb; pdb.set_trace()  # for debugging
            preds.append(postprocess(text))


        # 6) record results
        for ex, pred in zip(batch, preds):
            is_correct = pred.lower() == ex["object"].lower()
            results.append({
                "correct" :       is_correct,
                "sentence":       ex["query"],
                "entity":           ex["object"],
                "subject":        ex["subject"],
                "relation":       ex["template"],
                "prediction":     pred,
                "candidates":     ex["candidates"],
            })
            if is_correct:
                correct += 1

    acc = correct / len(results)
    print(f"\nAccuracy: {acc:.3%}  ({correct}/{len(results)})")

    out_path = Path(__file__).parent / args.outfile
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved predictions → {out_path.resolve()}")

    print("Filter rows with entity named 'edison' ...")
    filtered = [r for r in results if r["entity"].lower() == "edison"]
    print(f"Found {len(filtered)} rows with entity 'edison'.")

    if debug:
        print("\n--- DEBUG: whole prompts saved---")
        for i, (text, debug_len) in enumerate(zip(debug_whole, debug_index)):
            print(f"Prompt {i}: {text}")
            print(text[debug_len:], "→", results[i]["prediction"])


def stats():
    ds = load_dataset("coastalcph/pararel_patterns", split="train")
    # print(f"Dataset size: {len(ds)}")
    # print("Sample rows:")
    # print(set(ds["template"]))

    #find edison
    edison_rows = [r for r in ds if "edison" in r["subject"].lower()]
    tmp = [r for r in ds if "edison" in r["object"].lower()]

    print(f"Found {len(edison_rows)} rows with subject containing 'edison'.")
    print(f"Found {len(tmp)} rows with object containing 'edison'.")

if __name__ == "__main__":
    main()
    # stats()
