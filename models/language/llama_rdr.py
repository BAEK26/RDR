"""
Wrapper that instruments **Meta-Llama-3-8B-Instruct** so it plays nicely with
the original vision-centric Relaxed-Decision-Region (RDR) code-base.

Main public API
---------------
LlamaRDR.instance2featconfig(input_ids: LongTensor [, attention_mask]) -> (feat, config)

* `feat`   : FloatTensor  [bs, hidden_size]  – hidden activations of target layer
* `config` : IntTensor    [bs, hidden_size]  – binary (0/1) sign of FFN activations
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaRDR(nn.Module):
    """
    Instrumented Llama-3 that exposes *feature* and *binary configuration*
    tensors similar to what the original RDR expected from a CNN.

    Parameters
    ----------
    model_name : str
        🤗 identifier, default = `meta-llama/Llama-3.1-8B-Instruct`.
    target_layer : int
        0-based transformer layer index whose activations we study.
    capture_seq_pos : int | None
        Token position to use.  `-1` (default) = **last** token.
        If `None`, averages across the *whole* sequence.
    dtype : torch.dtype
        Model weights dtype, fp16 by default.
    device_map : "auto" | dict
        Passed straight to `from_pretrained`.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        target_layer: int = 27,
        capture_seq_pos: int | None = -1,
        dtype: torch.dtype = torch.float16,
        device_map: str | dict = "auto",
        **hf_kwargs,
    ):
        super().__init__()

        # ---------- load model + tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            # attn_implementation="flash_attention_2",
            **hf_kwargs,
        )

        self.hidden_size: int = self.model.config.hidden_size
        self.tlayer: int = target_layer
        self.spos: int | None = capture_seq_pos  # which token position

        # buffers that will be populated by forward hooks
        self._buffer_feat: torch.Tensor | None = None
        self._buffer_conf: torch.Tensor | None = None

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """Attach forward hooks to capture activations & configurations."""
        block = self.model.model.layers[self.tlayer]  # type: ignore
        def _extract_hidden(x):
            return x[0] if isinstance(x, (tuple,list)) else x
        # 1) Hidden *feature*  — output of the entire transformer block
        def save_feat(_, __, output):
            output = _extract_hidden(output)  # output can be a tuple
            # output: [bs, seq_len, hidden_size]

            self._buffer_feat = output.detach()

        block.register_forward_hook(save_feat, prepend=False)

        # 2) Binary *configuration* — activations inside the FFN/MoE (mlp)
        def save_conf(_, __, output):
            output = _extract_hidden(output)  # output can be a tuple
            # output: [bs, seq_len, hidden_size]
            self._buffer_conf = (output > 0).int().detach()

        # Llama's feed-forward sub-module is called ".mlp"
        block.mlp.register_forward_hook(save_conf, prepend=False)  # type: ignore

    # ------------------------------------------------------------------
    # Forward proxy  (optional – rarely used directly for RDR)
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public RDR helper
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def instance2featconfig(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the model once (without gradients) and returns the captured
        feature+config tensors **for each sequence in the batch**.

        Both outputs have shape `[bs, hidden_size]`.
        """
        # Clear stale buffers
        self._buffer_feat = None
        self._buffer_conf = None

        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        if self._buffer_feat is None or self._buffer_conf is None:
            raise RuntimeError("Hooks did not fire – check layer indices!")
        h_feat = self._buffer_feat
        h_conf = self._buffer_conf
        if self.spos is not None:
            feat = h_feat.mean(dim=1)  # average over seq_len
            conf = (h_conf.float().mean(dim=1) > 0.5).int()
        else:
            feat = h_feat[:, self.spos, :]
            conf = h_conf[:, self.spos, :]

        return feat.detach(), conf.detach()

    @torch.inference_mode()
    def instance2featconfig_pos(self, input_ids, ent_pos):
        """
        ent_pos : LongTensor[bs]  – index of the entity token for each example
        """
        self._buffer_feat = None
        self._buffer_conf = None
        out = self.model(input_ids=input_ids, use_cache=False)
        # out.hidden_states is NOT returned unless you set output_hidden_states;
        # we still rely on the forward hook, so the tensors are captured

        # After hook: _buffer_feat/conf have shape [bs, seq_len, hidden]
        # We now gather the per-sample token we want
        bs = input_ids.shape[0]
        gather_idx = ent_pos.to(self._buffer_feat.device)          # [bs]
        feat   = self._buffer_feat[torch.arange(bs), gather_idx, :]
        config = self._buffer_conf[torch.arange(bs), gather_idx, :]

        return feat.detach(), config.detach()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def tokenize(self, texts: list[str] | str, **tok_kwargs):
        """
        Shortcut to the underlying tokenizer that already sets padding/
        truncation flags convenient for batched inference.
        """
        defaults = dict(return_tensors="pt", padding=True, truncation=True)
        defaults.update(tok_kwargs)
        return self.tokenizer(texts, **defaults)
if __name__ == "__main__":
    rdr = LlamaRDR(target_layer=27, capture_seq_pos=-1)  # last token
    toks = rdr.tokenize(["I love this movie!", "Terrible plot..."]).to("cuda")
    feat, conf = rdr.instance2featconfig(toks["input_ids"])
    print("feat shape:", feat.shape)     # (2, 4096)
    print("conf shape:", conf.shape)     # (2, 4096)
    print("binary ratio:", conf.float().mean().item())
