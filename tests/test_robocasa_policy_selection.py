from argparse import Namespace
from dataclasses import FrozenInstanceError

import pytest

from robots.robocasa.policy_selection import (
    resolve_policy_family,
    resolve_policy_options,
)
from rpent.planning import get_planning_profile


@pytest.mark.parametrize("family", ["xiaomi-mibot"])
def test_vla_is_orthogonal_to_default_profile(family):
    assert resolve_policy_family(Namespace(vla=family)) == family


def test_xiaomi_profile_rejects_conflicting_vla():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_policy_family(Namespace(
            planning_profile="robocasa-xiaomi-mibot-gated-v1", vla="pi05"))


def test_xiaomi_defaults_and_chunk_limit():
    args = Namespace(vla="xiaomi-mibot")
    assert resolve_policy_options(args) == "xiaomi-mibot"
    assert args.actions_per_policy_request == 16
    assert args.policy_dtype == "bfloat16"
    args.actions_per_policy_request = 17
    with pytest.raises(ValueError, match="16"):
        resolve_policy_options(args)


def test_xiaomi_spec_is_frozen_and_legacy_profile_matches():
    from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_POLICY_SPEC

    spec = XIAOMI_MIBOT_POLICY_SPEC
    with pytest.raises(FrozenInstanceError):
        spec.crop_ratio = 1.0
    profile = get_planning_profile("robocasa-xiaomi-mibot-gated-v1")
    for key in ("observation_history", "observation_interval", "crop_ratio",
                "generated_chunk_size", "native_action_shape"):
        assert getattr(profile, key) == getattr(spec, key)
    assert spec.preset_name == "robocasa365-pandaomron-xiaomi-mibot-v1"
    assert spec.instruction_suffix == " /no_cot"


@pytest.mark.parametrize("family", ["xiaomi-mibot"])
def test_cli_accepts_selection_and_guidance_opt_out(family):
    from rpent.cli.main import _build_argparser
    from rpent.envs import get_env_spec

    parser = _build_argparser()
    get_env_spec("robocasa").add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(["--env", "robocasa", "--task", "SlideDishwasherRack",
                              "--vla", family, "--no-forward-step-guidance"])
    assert args.vla == family
    assert args.no_forward_step_guidance is True
