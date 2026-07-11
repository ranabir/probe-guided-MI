"""Model adapter layer: TransformerLens and HuggingFace adapters with a common interface."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    def __init__(self, model_config, device: torch.device, dtype: torch.dtype):
        self.model_config = model_config
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.n_layers: Optional[int] = None
        self.d_model: Optional[int] = None

    @abstractmethod
    def load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer into self.model / self.tokenizer."""

    @abstractmethod
    def format_prompt_response(self, prompt: str, response: str) -> str:
        """Combine prompt + response into a single string for the model."""

    @abstractmethod
    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Return tokenized tensors on self.device."""

    @abstractmethod
    def forward_with_cache(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> Dict[str, np.ndarray]:
        """
        Run model and return activation dict.

        Required keys (all shape [N, n_layers, d_model]):
            "hidden_states"

        Optional keys (shape [N, n_layers, d_model]):
            "attn_out"
            "mlp_out"
        """

    @abstractmethod
    def compute_probe_gradients(
        self,
        texts: List[str],
        probe,
        target_layer: int,
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        """
        Compute gradient of probe(h_layer) w.r.t. hidden states at all layers.
        Returns array of shape [N, n_layers, d_model].
        """

    @abstractmethod
    def get_mean_activations(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        """Return mean hidden state per layer, shape [n_layers, d_model]."""

    def compute_logprob(self, prompt: str, continuation: str, max_length: int = 128) -> float:
        """Compute log P(continuation | prompt) = sum of per-token log probs."""
        raise NotImplementedError

    def unload(self) -> None:
        """Free GPU memory."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _extract_position(tensor: torch.Tensor, position: str, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Extract a single token position from [B, T, D] -> [B, D]."""
    if position == "final":
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # [B]
            idx = lengths.clamp(min=0).cpu()  # always cpu to match cached tensors
            return tensor.cpu()[torch.arange(tensor.size(0)), idx]
        return tensor[:, -1, :]
    if position == "first":
        return tensor[:, 0, :]
    if position == "mean":
        if attention_mask is not None:
            mask = attention_mask.cpu().unsqueeze(-1).float()
            t = tensor.cpu()
            return (t * mask).sum(1) / mask.sum(1)
        return tensor.mean(1)
    raise ValueError(f"Unknown token_position: {position!r}")


# ---------------------------------------------------------------------------
# TransformerLens adapter
# ---------------------------------------------------------------------------


class TransformerLensAdapter(BaseModelAdapter):
    """Adapter for models supported by TransformerLens (GPT-2, Pythia, etc.)."""

    def load_model_and_tokenizer(self) -> None:
        import transformer_lens

        model_name = self.model_config.hf_name
        logger.info("Loading TransformerLens model: %s", model_name)

        # Map gpt2-small to 'gpt2' for TransformerLens
        tl_name = model_name
        if model_name == "gpt2-small":
            tl_name = "gpt2"

        dtype_map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        dtype_str = dtype_map.get(self.dtype, "float32")

        self.model = transformer_lens.HookedTransformer.from_pretrained(
            tl_name,
            dtype=dtype_str,
            device=str(self.device),
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        logger.info("Loaded %s: n_layers=%d, d_model=%d", tl_name, self.n_layers, self.d_model)

    def format_prompt_response(self, prompt: str, response: str) -> str:
        return f"{prompt} {response}"

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def forward_with_cache(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> Dict[str, np.ndarray]:
        enc = self.tokenize(texts, max_length=max_length)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        hook_names_resid = [f"blocks.{i}.hook_resid_post" for i in range(self.n_layers)]
        hook_names_attn = [f"blocks.{i}.hook_attn_out" for i in range(self.n_layers)]
        hook_names_mlp = [f"blocks.{i}.hook_mlp_out" for i in range(self.n_layers)]

        cache = {}

        def make_hook(key):
            def hook_fn(val, hook):
                cache[key] = val.detach().cpu()
            return hook_fn

        hook_fns = []
        for name in hook_names_resid + hook_names_attn + hook_names_mlp:
            hook_fns.append((name, make_hook(name)))

        with self.model.hooks(fwd_hooks=hook_fns):
            self.model(input_ids)

        n = len(texts)
        resid_layers = []
        attn_layers = []
        mlp_layers = []

        for i in range(self.n_layers):
            for dest, names in [
                (resid_layers, hook_names_resid),
                (attn_layers, hook_names_attn),
                (mlp_layers, hook_names_mlp),
            ]:
                name = names[i]
                if name in cache:
                    h = _extract_position(cache[name], token_position, attention_mask)
                    dest.append(h.float().numpy())
                else:
                    dest.append(np.zeros((n, self.d_model), dtype=np.float32))

        result = {
            "hidden_states": np.stack(resid_layers, axis=1),  # [N, L, D]
        }
        if any(np.any(a) for a in attn_layers):
            result["attn_out"] = np.stack(attn_layers, axis=1)
        if any(np.any(a) for a in mlp_layers):
            result["mlp_out"] = np.stack(mlp_layers, axis=1)

        return result

    def compute_probe_gradients(
        self,
        texts: List[str],
        probe,
        target_layer: int,
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        """Gradient of probe score w.r.t. residual stream at every layer.

        Strategy: inject a requires_grad leaf at the embedding stage so the
        computation graph flows through all subsequent layer activations.
        Call retain_grad() on each layer output so gradients are kept.
        """
        enc = self.tokenize(texts, max_length=max_length)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        n = len(texts)

        all_hook_names = [f"blocks.{i}.hook_resid_post" for i in range(self.n_layers)]
        target_hook_name = f"blocks.{target_layer}.hook_resid_post"

        stored: Dict[str, torch.Tensor] = {}

        def embed_hook(val, hook):
            # Replace embedding output with a leaf that has requires_grad=True
            val_req = val.detach().float().requires_grad_(True)
            stored["__embed__"] = val_req
            return val_req

        def layer_hook(val, hook):
            # Keep tensor in graph and retain grad so backward populates .grad
            val_f = val.float()
            val_f.retain_grad()
            stored[hook.name] = val_f
            return val_f

        fwd_hooks = [("hook_embed", embed_hook)] + [
            (name, layer_hook) for name in all_hook_names
        ]

        # Run with grad enabled (no torch.no_grad context)
        with torch.enable_grad():
            self.model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

        target_act = stored.get(target_hook_name)
        if target_act is None:
            return np.zeros((n, self.n_layers, self.d_model), dtype=np.float32)

        # Compute probe score from target layer activation (keep graph)
        h_pos = _extract_position(target_act, token_position, attention_mask)
        score = _probe_forward(probe, h_pos)
        score.sum().backward()

        # Collect gradients — layers before target have grad via residual chain,
        # layers after target have zero grad (probe doesn't depend on them)
        grad_layers = []
        for i in range(self.n_layers):
            name = all_hook_names[i]
            act = stored.get(name)
            if act is not None and act.grad is not None:
                g = _extract_position(act.grad, token_position, attention_mask)
                grad_layers.append(g.float().detach().cpu().numpy())
            else:
                grad_layers.append(np.zeros((n, self.d_model), dtype=np.float32))

        return np.stack(grad_layers, axis=1)  # [N, L, D]

    def get_mean_activations(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        acts = self.forward_with_cache(texts, max_length=max_length, token_position=token_position)
        return acts["hidden_states"].mean(axis=0)  # [L, D]

    @torch.no_grad()
    def compute_logprob(self, prompt: str, continuation: str, max_length: int = 128,
                        normalize: bool = False) -> float:
        """log P(continuation | prompt) using TransformerLens.

        If normalize=True, returns mean per-token logprob (length-normalized).
        """
        full_text = prompt + " " + continuation
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_len = prompt_ids.shape[1]

        logits = self.model(full_ids)  # [1, T, V]
        log_probs = torch.log_softmax(logits[0].float().cpu(), dim=-1)

        continuation_ids = full_ids[0, prompt_len:].cpu()
        n_tok = continuation_ids.numel()
        if n_tok == 0:
            return 0.0
        positions = torch.arange(prompt_len - 1, full_ids.shape[1] - 1)
        total = float(log_probs[positions, continuation_ids].sum().item())
        return total / n_tok if normalize else total


# ---------------------------------------------------------------------------
# HuggingFace adapter
# ---------------------------------------------------------------------------


class HuggingFaceAdapter(BaseModelAdapter):
    """Adapter for models loaded via HuggingFace transformers (Qwen, Gemma, etc.)."""

    def load_model_and_tokenizer(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.model_config.hf_name
        logger.info("Loading HuggingFace model: %s", model_name)

        torch_dtype = self.dtype if self.dtype != torch.float32 else "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if str(self.device) != "mps":
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Detect architecture
        self._detect_architecture()
        logger.info(
            "Loaded %s: n_layers=%d, d_model=%d", model_name, self.n_layers, self.d_model
        )

    def _detect_architecture(self) -> None:
        """Try to find the number of layers and d_model from model config."""
        cfg = self.model.config
        self.n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 12
        self.d_model = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", None)
            or 768
        )

    def _get_layers(self):
        """Return the list of transformer layer modules."""
        # Walk the layer_attr path (e.g. "model.layers")
        obj = self.model
        for part in self.model_config.layer_attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None or not hasattr(obj, "__len__"):
            # Fallback: look for common attribute names
            for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
                try:
                    sub = self.model
                    for part in attr.split("."):
                        sub = getattr(sub, part)
                    if hasattr(sub, "__len__"):
                        return sub
                except AttributeError:
                    continue
        return obj

    def format_prompt_response(self, prompt: str, response: str) -> str:
        if self.model_config.chat_template and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                pass
        return f"{prompt} {response}"

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def forward_with_cache(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> Dict[str, np.ndarray]:
        enc = self.tokenize(texts, max_length=max_length)
        attention_mask = enc.get("attention_mask")

        hidden_cache: List[torch.Tensor] = []
        attn_cache: List[torch.Tensor] = []
        mlp_cache: List[torch.Tensor] = []
        hooks = []
        layers = self._get_layers()

        def make_layer_hook(store):
            def hook_fn(module, input, output):
                # output can be a tuple; first element is hidden state
                h = output[0] if isinstance(output, tuple) else output
                store.append(h.detach().cpu())
            return hook_fn

        def make_sub_hook(store):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                store.append(h.detach().cpu())
            return hook_fn

        for layer in layers:
            hooks.append(layer.register_forward_hook(make_layer_hook(hidden_cache)))
            # Try sub-component hooks
            if hasattr(layer, "self_attn"):
                hooks.append(layer.self_attn.register_forward_hook(make_sub_hook(attn_cache)))
            if hasattr(layer, "mlp"):
                hooks.append(layer.mlp.register_forward_hook(make_sub_hook(mlp_cache)))

        try:
            outputs = self.model(**enc, output_hidden_states=False)
        finally:
            for h in hooks:
                h.remove()

        n = len(texts)

        def stack_cache(cache, n_layers_expected):
            if not cache or len(cache) != n_layers_expected:
                return None
            stacked = []
            for h in cache:
                pos = _extract_position(h, token_position, attention_mask)
                stacked.append(pos.float().numpy())
            return np.stack(stacked, axis=1)  # [N, L, D]

        result = {}
        hs = stack_cache(hidden_cache, self.n_layers)
        if hs is not None:
            result["hidden_states"] = hs

        if not result:
            # Fallback: use output_hidden_states
            logger.warning("Hook-based caching failed; using output_hidden_states fallback.")
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            all_hs = out.hidden_states  # tuple of [B, T, D], length = n_layers + 1
            stacked = []
            for h in all_hs[1:]:  # skip embedding layer
                pos = _extract_position(h.cpu(), token_position, attention_mask)
                stacked.append(pos.float().numpy())
            result["hidden_states"] = np.stack(stacked, axis=1)

        attn = stack_cache(attn_cache, self.n_layers)
        if attn is not None:
            result["attn_out"] = attn

        mlp = stack_cache(mlp_cache, self.n_layers)
        if mlp is not None:
            result["mlp_out"] = mlp

        return result

    def compute_probe_gradients(
        self,
        texts: List[str],
        probe,
        target_layer: int,
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        """Compute gradient of probe score w.r.t. hidden states at all layers.

        Strategy: inject a leaf tensor with requires_grad=True at the first layer
        output so the computation graph flows through all subsequent layers.
        Call retain_grad() on each layer output so .grad is populated after backward.
        """
        enc = self.tokenize(texts, max_length=max_length)
        attention_mask = enc.get("attention_mask")
        layers = self._get_layers()
        n = len(texts)

        stored: Dict[str, torch.Tensor] = {}
        hooks = []
        layer_list = list(layers)

        def make_first_layer_hook(idx):
            """For the first layer: detach and create a leaf with requires_grad."""
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                # Detach from model graph and create a leaf that owns the gradient
                leaf = h.detach().float().requires_grad_(True)
                stored[idx] = leaf
                if isinstance(output, tuple):
                    return (leaf,) + output[1:]
                return leaf
            return hook_fn

        def make_layer_hook(idx):
            """For subsequent layers: retain grad so .grad is populated after backward."""
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                h_f = h.float()
                h_f.retain_grad()
                stored[idx] = h_f
                if isinstance(output, tuple):
                    return (h_f,) + output[1:]
                return h_f
            return hook_fn

        for i, layer in enumerate(layer_list):
            if i == 0:
                hooks.append(layer.register_forward_hook(make_first_layer_hook(0)))
            else:
                hooks.append(layer.register_forward_hook(make_layer_hook(i)))

        try:
            for p in self.model.parameters():
                p.requires_grad_(False)
            with torch.enable_grad():
                self.model(**enc)
        finally:
            for h in hooks:
                h.remove()

        if len(stored) != self.n_layers:
            logger.warning("Gradient hook captured %d/%d layers; returning zeros.", len(stored), self.n_layers)
            return np.zeros((n, self.n_layers, self.d_model), dtype=np.float32)

        h_target = stored.get(target_layer)
        if h_target is None:
            return np.zeros((n, self.n_layers, self.d_model), dtype=np.float32)

        h_pos = _extract_position(h_target, token_position, attention_mask)
        score = _probe_forward(probe, h_pos)
        score.sum().backward()

        grad_layers = []
        for i in range(self.n_layers):
            h = stored.get(i)
            if h is not None and h.grad is not None:
                g = _extract_position(h.grad, token_position, attention_mask)
                grad_layers.append(g.float().detach().cpu().numpy())
            else:
                grad_layers.append(np.zeros((n, self.d_model), dtype=np.float32))

        return np.stack(grad_layers, axis=1)

    def get_mean_activations(
        self,
        texts: List[str],
        max_length: int = 128,
        token_position: str = "final",
    ) -> np.ndarray:
        acts = self.forward_with_cache(texts, max_length=max_length, token_position=token_position)
        return acts["hidden_states"].mean(axis=0)  # [L, D]

    @torch.no_grad()
    def compute_logprob(self, prompt: str, continuation: str, max_length: int = 128,
                        normalize: bool = False) -> float:
        """log P(continuation | prompt) using HuggingFace model.

        If normalize=True, returns mean per-token logprob (length-normalized).
        """
        full_text = prompt + " " + continuation
        full_enc = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        prompt_enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_len = prompt_enc["input_ids"].shape[1]

        outputs = self.model(**full_enc)
        log_probs = torch.log_softmax(outputs.logits[0].float().cpu(), dim=-1)  # [T, V]

        input_ids = full_enc["input_ids"][0].cpu()
        continuation_ids = input_ids[prompt_len:]
        n_tok = continuation_ids.numel()
        if n_tok == 0:
            return 0.0
        positions = torch.arange(prompt_len - 1, input_ids.shape[0] - 1)
        total = float(log_probs[positions, continuation_ids].sum().item())
        return total / n_tok if normalize else total


# ---------------------------------------------------------------------------
# Probe forward helper (sklearn probe -> torch score)
# ---------------------------------------------------------------------------


def _probe_forward(probe, h: torch.Tensor) -> torch.Tensor:
    """Differentiable probe output from a fitted LinearProbe.

    - regression: linear output (the predicted target, e.g. behavior_margin)
    - classification: sigmoid / softmax probability of the positive class
    """
    coef = torch.tensor(np.asarray(probe.coef_), dtype=torch.float32)
    bias = torch.tensor(np.asarray(probe.intercept_), dtype=torch.float32).reshape(-1)
    task = getattr(probe, "task", "classification")

    if task == "regression":
        w = coef.reshape(-1)  # [D]
        return h @ w + bias[0]  # [B]

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[0] == 1:
        logit = h @ coef.T + bias  # [B, 1]
        return torch.sigmoid(logit.squeeze(-1))  # [B]
    logit = h @ coef.T + bias  # [B, C]
    return torch.softmax(logit, dim=-1)[:, 1]
