"""SAM3 model for fine-tuning: build, load sam3.pt (checked), freeze per configuration, save / load
the trained part.

Configurations (decided 17/11; parameter names = Sam3Image state dict without the "detector." prefix):
  A2 (main)    : train everything after the image trunk and the text encoder
                 = neck convs, text resizer, fusion encoder, decoder, dot-product scoring,
                   segmentation head (about 32.6M parameters, 3.9%)
  A1 (ablation): train the segmentation head only (about 2.3M, 0.27%)
  base         : nothing trained (SAM3 as released), used by the preflight check
Frozen in both: image trunk (447M), text encoder (354M), geometry encoder (8M; text prompts only).

Frozen blocks are kept in eval() mode even when the Trainer calls model.train(): no drop-path /
dropout noise on frozen features, and no activation checkpointing inside them (no gradient flows there).
The repo's freeze_* YAML keys are not read by any code, so freezing is done here with requires_grad.
"""
from __future__ import annotations

import types
from pathlib import Path

import torch

import ft_common as fc

# trained parameter-name prefixes per configuration
TRAIN_PREFIXES = {
    "A2": ["backbone.vision_backbone.convs.", "backbone.language_backbone.resizer.", "transformer.encoder.",
           "transformer.decoder.", "dot_prod_scoring.", "segmentation_head."],
    "A1": ["segmentation_head."],
    "base": [],
}
# every parameter of the model must fall in exactly one of these groups (otherwise we stop: nothing
# is frozen or trained by accident)
KNOWN_GROUPS = ["backbone.vision_backbone.trunk.", "backbone.language_backbone.encoder.",
                "backbone.vision_backbone.convs.", "backbone.language_backbone.resizer.",
                "transformer.encoder.", "transformer.decoder.", "dot_prod_scoring.", "segmentation_head.",
                "geometry_encoder."]
# expected trainable parameter counts (counted from sam3.pt, 17/11), checked with a 3% margin
EXPECTED_TRAINABLE = {"A2": 32.6e6, "A1": 2.3e6, "base": 0}


def group_of(name):
    g = [p for p in KNOWN_GROUPS if name.startswith(p)]
    if len(g) != 1:
        raise fc.PipelineError(f"parameter {name} is in {len(g)} known groups (expected 1); update ft_model.KNOWN_GROUPS")
    return g[0]


def frozen_groups(cfg):
    return [g for g in KNOWN_GROUPS if g not in TRAIN_PREFIXES[cfg]]


def frozen_patterns(cfg):
    """unix patterns for Trainer.checkpoint.skip_saving_parameters (frozen weights are not saved;
    they come back from sam3.pt when the model is rebuilt)."""
    return [g + "*" for g in frozen_groups(cfg)]


def bpe_path():
    import sam3
    p = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    fc.need(p, "SAM3 BPE vocabulary")
    return str(p)


def load_sam3_weights(model, checkpoint_path):
    """Load sam3.pt into a Sam3Image and CHECK it (the repo's own loader only prints missing keys)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sd = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}   # = model_builder._load_checkpoint
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise fc.PipelineError(f"sam3.pt is missing {len(missing)} model keys, e.g. {missing[:5]}")
    # sam2_convs belong to the instance-interactivity neck, which the image model does not build
    unexpected = [k for k in unexpected if not k.startswith("backbone.vision_backbone.sam2_convs.")]
    if unexpected:
        raise fc.PipelineError(f"sam3.pt has {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    return model


def _build(checkpoint_path, eval_mode):
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(bpe_path=bpe_path(), device="cpu", eval_mode=eval_mode,
                                   checkpoint_path=None, load_from_HF=False, enable_segmentation=True)
    return load_sam3_weights(model, checkpoint_path)


def apply_freeze(model, cfg):
    """requires_grad per configuration + keep frozen blocks in eval mode. Returns a parameter report."""
    assert cfg in TRAIN_PREFIXES, cfg
    n_train = n_total = 0
    per_group = {}
    for name, p in model.named_parameters():
        g = group_of(name)
        train = g in TRAIN_PREFIXES[cfg]
        p.requires_grad_(train)
        n_total += p.numel()
        n_train += p.numel() if train else 0
        d = per_group.setdefault(g.rstrip("."), {"params": 0, "trained": train})
        d["params"] += p.numel()
    exp = EXPECTED_TRAINABLE[cfg]
    if exp and abs(n_train - exp) > 0.03 * exp:
        raise fc.PipelineError(f"{cfg}: {n_train / 1e6:.2f}M trainable parameters, expected ~{exp / 1e6:.1f}M")
    if not exp and n_train:
        raise fc.PipelineError(f"{cfg}: {n_train} trainable parameters, expected 0")

    frozen_modules = [g.rstrip(".") for g in frozen_groups(cfg)]
    base_train = type(model).train

    def train(self, mode=True):                 # nn.Module.train recursion calls this on our model
        base_train(self, mode)
        for m in frozen_modules:
            self.get_submodule(m).eval()
        return self

    model.train = types.MethodType(train, model)
    model.train(model.training)

    # The image trunk's MLP always runs the fused inference kernel sam3.perflib.fused.addmm_act, which raises
    # "Expected grad to be disabled" when autograd is on (repo commit f1647e8: the trunk cannot be trained).
    # The trunk is frozen in every configuration and its input is the image, so no gradient has to flow
    # through it: run its forward under no_grad (same values; the trained neck convs get the output as a
    # constant input).
    trunk = model.backbone.vision_backbone.trunk
    assert not any(p.requires_grad for p in trunk.parameters()), "the image trunk must be frozen"
    trunk_forward = trunk.forward

    def trunk_no_grad(*args, **kwargs):
        with torch.no_grad():
            return trunk_forward(*args, **kwargs)

    trunk.forward = trunk_no_grad
    return {"config": cfg, "trainable": n_train, "total": n_total, "trainable_share": n_train / n_total,
            "groups": per_group, "frozen_eval_modules": frozen_modules}


def build_ft_model(ft_config, checkpoint_path, **_ignored):
    """Hydra target of trainer.model (configs/ft_sam3_seg.yaml): training model for A1 / A2."""
    model = _build(checkpoint_path, eval_mode=False)       # eval_mode=False -> the repo adds its matcher
    rep = apply_freeze(model, ft_config)
    print(f"[ft_model] {ft_config}: {rep['trainable'] / 1e6:.2f}M / {rep['total'] / 1e6:.1f}M parameters trained "
          f"({100 * rep['trainable_share']:.2f}%); frozen in eval mode: {rep['frozen_eval_modules']}", flush=True)
    model.ft_report = rep
    model.ft_config = ft_config
    return model


def trainable_state_dict(model):
    """Only the trained parameters (the 'patch' on top of sam3.pt), on CPU."""
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def save_best(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state": trainable_state_dict(model), "meta": meta}, tmp)
    tmp.replace(path)


def load_best_into(model, best_path, cfg):
    """Overlay a best checkpoint on a model built from sam3.pt; checks every trained key is present."""
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    state, meta = ck["state"], ck["meta"]
    if meta.get("config") != cfg:
        raise fc.PipelineError(f"{best_path}: checkpoint of config {meta.get('config')}, expected {cfg}")
    want = {n for n, _ in model.named_parameters() if group_of(n) in TRAIN_PREFIXES[cfg]}
    if set(state) != want:
        raise fc.PipelineError(f"{best_path}: {len(set(state) ^ want)} keys differ from the {cfg} trained set")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing).isdisjoint(want)
    return meta


def build_eval_model(cfg, checkpoint_path, best_path=None, device="cuda"):
    """Model for inference: sam3.pt (+ best checkpoint of cfg), eval mode, on device."""
    model = _build(checkpoint_path, eval_mode=True)
    meta = None
    if cfg != "base":
        if best_path is None:
            raise fc.PipelineError(f"{cfg}: best checkpoint required")
        meta = load_best_into(model, best_path, cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device).eval()
    return model, meta
