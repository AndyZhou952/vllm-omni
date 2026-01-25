# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.lora.utils import _expand_expected_modules_for_merged_projections


def test_expand_expected_modules_for_packed_projections():
    supported = {"to_qkv", "w13", "some_other"}
    expanded = _expand_expected_modules_for_merged_projections(supported)

    # Keep originals.
    assert supported.issubset(expanded)

    # Expand packed projections to their logical submodules.
    assert {"to_q", "to_k", "to_v"}.issubset(expanded)
    assert {"w1", "w3"}.issubset(expanded)


def test_expand_expected_modules_noop_without_packed_modules():
    supported = {"to_q", "to_k", "to_v"}
    expanded = _expand_expected_modules_for_merged_projections(supported)
    assert expanded == supported
