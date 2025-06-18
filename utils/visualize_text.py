"""
Light-weight console visualiser for RDR over language models.

It prints a handful of samples that fall inside the Relaxed Decision Region
together with their gold label and (optionally) the model’s prediction.

-------------------------------------------------------------------------
Example
-------
>>> from utilities.visualize_text import visualize_text
>>> visualize_text(rdr_samples, dataset, k=5, pred_fn=my_predict)
-------------------------------------------------------------------------

If `termcolor` is installed the label/pred are colourised; otherwise plan text.
"""

from __future__ import annotations
import random
from typing import Callable, Sequence

try:
    from termcolor import colored
except ImportError:  # graceful fallback
    def colored(text, *_args, **_kwargs):
        return str(text)


def _fmt_label(lbl: int, *, is_pred: bool = False, correct: bool | None = None):
    if correct is None:
        return f"{lbl}"
    color = "green" if correct else "red"
    tag   = "pred" if is_pred else "gold"
    return colored(f"{tag}:{lbl}", color)


# ---------------------------------------------------------------------
def visualize_text(
    sample_indices: Sequence[int],
    dataset,
    *,
    k: int = 10,
    pred_fn: Callable[[str], int] | None = None
) -> None:
    """
    Print up to *k* random samples from `sample_indices`.

    Parameters
    ----------
    sample_indices : Sequence[int]
        Indices (into *dataset*) returned by `RDR.selection(...)`.
    dataset : torch.utils.data.Dataset
        Must yield dict with `'prompt'` and `'true_labels'` or `'label'`.
    k : int
        How many examples to print.
    pred_fn : callable | None
        If provided, called as `pred_fn(prompt:str) -> int` to produce
        a model prediction that will be compared with the gold label.
    """
    if len(sample_indices) == 0:
        print("⛔ No samples to show.")
        return

    # choose subset without replacement
    chosen = random.sample(sample_indices, min(k, len(sample_indices)))

    print(f"\n=== Visualising {len(chosen)} / {len(sample_indices)} samples ===")

    for idx in chosen:
        item   = dataset[idx]
        prompt = item["prompt"]
        gold   = item.get("true_labels", item.get("label", None))

        # compute pred if fn provided
        if pred_fn is not None:
            pred   = pred_fn(prompt)
            correct = (pred == gold)
            gold_s  = _fmt_label(gold, correct=True)
            pred_s  = _fmt_label(pred, is_pred=True, correct=correct)
            header  = f"[{idx}] {gold_s} | {pred_s}"
        else:
            header  = f"[{idx}] label:{gold}"

        print("-" * 80)
        print(header)
        print(prompt.strip())
    print("-" * 80)
    print("End of RDR visualisation\n")

def visualize_entity(samples, dataset, print_max=10):
    for idx in random.sample(samples, min(print_max, len(samples))):
        row  = dataset[idx]
        sent = row["sentence"]
        span = row["char_span"]
        tok  = row["entity"]

        # DEBUG
        # print("-" * 40)
        # print(f"idx={idx}  entity='{tok}'  span={span}")
        # print("sentence:", sent)

        # old colouring
        coloured = sent[:span[0]] + "\033[1;34m" + sent[span[0]:span[1]] + \
                   "\033[0m" + sent[span[1]:]
        print(coloured)