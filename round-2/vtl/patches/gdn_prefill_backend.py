"""Env-driven pin of the GDN prefill backend (ported from round-1.1, minus the chunk-scan
scaffold that round-2 does not carry).

vLLM v0.25.0 picks the gated-delta-rule prefill kernel in
``qwen_gdn_linear_attn._resolve_gdn_prefill_backend`` from
``--additional-config '{"gdn_prefill_backend": ...}'``: FlashInfer on Hopper (SM90) by
default, the in-tree CuteDSL kernel on Blackwell opt-in, Triton/FLA otherwise. Rebuilding
the serve command to A/B those three costs a compose edit per arm; this patch reads

    VTL_GDN_PREFILL_BACKEND = auto | triton | flashinfer | cutedsl      (default: auto)

and, when it is not ``auto``, injects the choice into ``additional_config`` around the
stock resolver call. DELIBERATELY a pin of the *requested* backend, not of the resolved
method: every stock support guard still runs (FlashInfer's SM90/SM100 gates, the
cutlass-dsl install-integrity check, CuteDSL's Blackwell-only gate), so a backend the
device cannot run degrades to Triton with vLLM's own warning instead of crashing the
server -- on the H200 (SM90) that means ``cutedsl`` resolves to triton, exactly as a
--additional-config request would.

``auto`` (or unset) leaves stock resolution completely untouched -- the wrapper is not
even installed. Registry: ``gdn_prefill_backend``, default OFF
(``VTL_ENABLE_GDN_PREFILL_BACKEND=1`` to arm the patch, then the env var above per arm).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.gdn_prefill_backend")

_CHOICES = ("auto", "triton", "flashinfer", "cutedsl")


def _backend_choice() -> str:
    return os.environ.get("VTL_GDN_PREFILL_BACKEND", "auto").strip().lower()


def _call_with_pin(orig, vllm_config, choice: str):  # noqa: ANN001
    """Call the stock resolver with ``gdn_prefill_backend`` pinned to ``choice`` in
    ``additional_config``, restoring the config afterwards. Pure enough to self-check:
    everything it needs from vllm_config is one dict-valued attribute."""
    ac = getattr(vllm_config, "additional_config", None)
    if not isinstance(ac, dict):
        # v0.25.0's resolver treats a non-dict additional_config as "auto"; with no dict
        # to inject into there is nothing safe to pin -- leave stock.
        log.warning("vtl: additional_config is %s, cannot pin gdn prefill backend",
                    type(ac).__name__)
        return orig(vllm_config)
    had_key = "gdn_prefill_backend" in ac
    saved = ac.get("gdn_prefill_backend")
    ac["gdn_prefill_backend"] = choice
    try:
        return orig(vllm_config)
    finally:
        if had_key:
            ac["gdn_prefill_backend"] = saved
        else:
            ac.pop("gdn_prefill_backend", None)


@register_patch("gdn_prefill_backend", default=False)
def apply() -> None:
    choice = _backend_choice()
    if choice in ("", "auto"):
        return  # nothing to force; leave stock vLLM entirely alone
    if choice not in _CHOICES:
        log.warning("vtl: VTL_GDN_PREFILL_BACKEND=%r not in %s; leaving stock",
                    choice, list(_CHOICES))
        return

    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as mod
    except Exception as exc:  # module moved/renamed in this vLLM -> stay stock
        log.warning("vtl: qwen_gdn_linear_attn not importable (%s); backend unchanged", exc)
        return
    if not hasattr(mod, "_resolve_gdn_prefill_backend"):
        log.warning("vtl: _resolve_gdn_prefill_backend absent; backend unchanged")
        return
    if already_patched(mod, "_resolve_gdn_prefill_backend"):
        return

    orig = mod._resolve_gdn_prefill_backend

    def _resolve(vllm_config):  # noqa: ANN001
        current = _backend_choice()
        if current in ("", "auto") or current not in _CHOICES:
            return orig(vllm_config)
        try:
            requested, active = _call_with_pin(orig, vllm_config, current)
        except Exception as exc:  # any drift -> stock resolution, never crash the server
            log.warning("vtl: gdn prefill pin failed (%s); stock resolution", exc)
            return orig(vllm_config)
        log.info("vtl: GDN prefill backend pinned: requested=%s -> active=%s",
                 requested, active)
        return requested, active

    mod._resolve_gdn_prefill_backend = mark_patched(_resolve, orig)
    log.info("vtl: GDN prefill backend pin installed (VTL_GDN_PREFILL_BACKEND=%s)", choice)


def _self_check() -> None:
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "gdn_prefill_backend")
    assert patch.default is False, "prefill pin is an A/B lever, off by default"
    assert is_enabled(patch) is False
    os.environ["VTL_ENABLE_GDN_PREFILL_BACKEND"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_PREFILL_BACKEND")

    os.environ.pop("VTL_GDN_PREFILL_BACKEND", None)
    assert _backend_choice() == "auto"
    os.environ["VTL_GDN_PREFILL_BACKEND"] = " FlashInfer "
    assert _backend_choice() == "flashinfer"
    os.environ.pop("VTL_GDN_PREFILL_BACKEND")

    # apply() with auto/unset must be a no-op even without vLLM importable.
    apply()

    # The pin injects, the resolver sees it, and the config is restored afterwards --
    # including the "key already present" case, which must not be clobbered.
    class _Cfg:
        def __init__(self, ac):  # noqa: ANN001
            self.additional_config = ac

    seen = []

    def fake_resolver(cfg):  # noqa: ANN001
        seen.append(cfg.additional_config.get("gdn_prefill_backend"))
        return cfg.additional_config.get("gdn_prefill_backend"), "triton"

    cfg = _Cfg({})
    assert _call_with_pin(fake_resolver, cfg, "flashinfer") == ("flashinfer", "triton")
    assert seen == ["flashinfer"]
    assert "gdn_prefill_backend" not in cfg.additional_config, "injected key must be removed"

    cfg2 = _Cfg({"gdn_prefill_backend": "cutedsl", "other": 1})
    assert _call_with_pin(fake_resolver, cfg2, "triton") == ("triton", "triton")
    assert cfg2.additional_config["gdn_prefill_backend"] == "cutedsl", "must restore prior value"
    assert cfg2.additional_config["other"] == 1

    # Non-dict additional_config: pin refuses and calls stock as-is.
    class _Odd:
        additional_config = None

    calls = []
    assert _call_with_pin(lambda c: calls.append(c) or ("auto", "triton"), _Odd(), "triton") \
        == ("auto", "triton")
    assert len(calls) == 1

    # A raising resolver inside the pin path must not corrupt the config.
    def boom(cfg):  # noqa: ANN001
        raise RuntimeError("drift")

    cfg3 = _Cfg({})
    try:
        _call_with_pin(boom, cfg3, "flashinfer")
    except RuntimeError:
        pass
    assert "gdn_prefill_backend" not in cfg3.additional_config

    print("gdn_prefill_backend self-check ok")


if __name__ == "__main__":
    _self_check()
