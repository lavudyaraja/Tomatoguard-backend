import timm


# ── Preferred NextViT model names (in order of priority) ─────────────────────
_NEXTVIT_PREFERENCES = [
    "nextvit_small",
    "nextvit_base",
    "nextvit_large",
    "nextvit_small.in1k",
    "nextvit_base.in1k",
    "nextvit_large.in1k",
]


def _pick_nextvit_name() -> str:
    """Return the first available NextViT model name from timm."""
    available = set(timm.list_models("nextvit*"))
    for name in _NEXTVIT_PREFERENCES:
        if name in available:
            return name
    # Fallback: grab whatever is available
    fallback = timm.list_models("nextvit*")
    if fallback:
        return sorted(fallback)[0]
    raise RuntimeError(
        "No NextViT model found in timm. Run: pip install --upgrade timm"
    )


def create_nextvit(
    num_classes: int = 11,
    dropout: float = 0.2,
    drop_path_rate: float = 0.1,
    pretrained: bool = False,
    model_name: str = None,
):
    """
    Factory that creates a NextViT model via timm.

    Parameters mirror the training notebook's build_model() function so that
    the checkpoint's model_state_dict loads without key mismatches.

    Args:
        num_classes:      Number of output classes (11 for TomatoGuard).
        dropout:          Classifier head drop-rate.
        drop_path_rate:   Stochastic-depth drop-rate used during training.
        pretrained:       Load ImageNet-pretrained weights from timm.
                          Set to False when you will load a fine-tuned checkpoint.
        model_name:       Override the auto-selected model name.

    Returns:
        A timm model ready for inference.
    """
    name = model_name or _pick_nextvit_name()
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=drop_path_rate,
    )
    return model


if __name__ == "__main__":
    m = create_nextvit(pretrained=False)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"NextViT OK: {type(m).__name__} | Params: {total:.1f}M")
