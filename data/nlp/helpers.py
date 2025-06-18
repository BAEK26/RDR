from typing import Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


# ------------------------------------------------------------------
# Shared tokenizer (used both for char→token alignment and later batching)
# ------------------------------------------------------------------
_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

# ------------------------------------------------------------------
# Helper: char-span → first sub-token index
# ------------------------------------------------------------------
def char_to_token(sentence: str, span: Tuple[int, int]) -> int:
    """
    Return the index of the first *sub-token* whose [c0, c1) interval
    overlaps the entity start.  Raises ValueError if we still can't find one.
    """
    start, _ = span
    enc = _tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    for tidx, (c0, c1) in enumerate(enc.offset_mapping):
        # Llama/SentencePiece may mark special tokens with (-1, -1)
        if c0 == c1 == 0:
            continue
        if c0 <= start < c1:          # overlap test instead of equality
            return tidx

    # fallback: pick the first token that *starts after* the entity
    for tidx, (c0, _c1) in enumerate(enc.offset_mapping):
        if c0 >= start:
            return tidx

    raise ValueError("Entity span not aligned with tokenizer offsets.")
