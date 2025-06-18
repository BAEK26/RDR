"""
WikiNeural (English) wrapper for entity-centric RDR.

Each sample dict contains
    sentence          : str
    entity            : str                 # first entity mention
    char_span         : (start, end)        # inclusive/exclusive char offsets
    entity_token_idx  : int                 # index of FIRST sub-token
    label             : str                 # PER / ORG / LOC / MISC
"""

from __future__ import annotations
from typing import Literal, Dict, Any

from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset
from data.nlp.helpers import char_to_token, _tokenizer





# ------------------------------------------------------------------
class WikiNeuralEnt(Dataset):
    _NER_ID2TAG = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    def __init__(
        self,
        split: Literal["train_en", "validation_en", "test_en"] = "test_en",
    ):
        """
        `split` must end with `_en` to mirror WikiNeural's English subsets.
        """
        if not split.endswith("_en"):
            raise ValueError("split must be like 'test_en', 'train_en', ...")

        hf_split = split.replace("_en", "")  # 'train', 'validation', 'test'
        self.ds = load_dataset("Babelscape/wikineural", split=split)

    # -------------------------
    def __len__(self) -> int:
        return len(self.ds)

    # -------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        tokens: list[str] = row["tokens"]
        tag_ids: list[int] = row["ner_tags"]

        # --- convert to string tags once ---
        tags = [self._NER_ID2TAG[i] for i in tag_ids]

        # --- find first entity token ---
        ent_tok_idx, ent_type = None, None
        for i, tag in enumerate(tags):
            if tag != "O":                      # first non-O
                ent_tok_idx = i
                ent_type = tag.split("-")[-1]   # B-PER → PER
                break
        if ent_tok_idx is None:
            raise ValueError("Sentence contains no entity — filter beforehand.")

        entity_word = tokens[ent_tok_idx]

        # --- build plain sentence string ---
        # join without space before punctuation to mimic natural text
        PUNCT = {".", ",", ";", ":", "?", "!", "'s", "'m", "'re", "'ve", "'ll", ")", "]", "”", "'"}
        SUFFIX = {"[", "(", "“", "‘", "«", "‹", "【", "〔", "〖", "『", "（", "［", "｛", "｢", "『", "「"}
        sentence_tokens = []
        char_start = None  # start position of the entity in the sentence
        for i, tok in enumerate(tokens):
            # record start position *just before* we append a space
            if i == ent_tok_idx and char_start is None:
                char_start = sum(len(s) for s in sentence_tokens) + len(sentence_tokens)  # +1 for spaces
                if i!= 0 and sentence_tokens[-1] in SUFFIX:
                    char_start -= 1

            if i == 0:
                sentence_tokens.append(tok)
            elif sentence_tokens[-1] in SUFFIX:
                sentence_tokens[-1] += tok
            elif tok in PUNCT:
                sentence_tokens[-1] += tok
            else:
                sentence_tokens.append(tok)



        sentence = " ".join(sentence_tokens)
        char_end = char_start + len(entity_word)

        # --- map char span → llama sub-token index ---
        ent_tok_id = char_to_token(sentence, (char_start, char_end))
        try:
            assert entity_word.strip() == sentence[char_start:char_end].strip()
            assert _tokenizer.decode(_tokenizer.encode(sentence, add_special_tokens=False)[ent_tok_id]).strip() in entity_word.strip()
        except AssertionError as e:
            print("entity_word:", entity_word)

        return {
            "sentence":          sentence,
            "entity":            entity_word,
            "char_span":         (char_start, char_end),
            "entity_token_idx":  ent_tok_id,
            "label":             ent_type,       # PER / ORG / LOC / MISC
        }


# ------------------------------------------------------------------
def get_split(split: str = "test_en") -> WikiNeuralEnt:
    """
    Convenience factory mirroring other dataset helpers.

    Examples
    --------
    >>> ds = get_split("test_en")
    >>> sample = ds[0]
    >>> sample["entity"], sample["label"]
    ('Brad', 'PER')
    """
    return WikiNeuralEnt(split=split)
