"""Select a VLA independently of the Harness planning algorithm."""

from rpent.planning import ROBOCASA_HORIZON_PLANNING_PROFILE, get_planning_profile
from rpent.policies.robocasa_pi05_catalog import PI05_CHECKPOINT
from rpent.policies.smolvla_catalog import SMOLVLA_CHECKPOINTS
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_MIBOT_POLICY_SPEC,
)


def resolve_policy_family(args):
    profile = get_planning_profile(
        getattr(args, "planning_profile", None) or ROBOCASA_HORIZON_PLANNING_PROFILE
    )
    selected = getattr(args, "vla", None)
    if profile.policy_family == "xiaomi-mibot":
        if selected not in {None, "xiaomi-mibot"}:
            raise ValueError(
                "Xiaomi planning profile cannot be combined with another --vla"
            )
        return "xiaomi-mibot"
    if selected not in {None, "pi05", "smolvla", "xiaomi-mibot"}:
        raise ValueError(f"unsupported RoboCasa VLA: {selected!r}")
    return selected or "smolvla"


def policy_checkpoint(family):
    return {
        "pi05": PI05_CHECKPOINT,
        "smolvla": SMOLVLA_CHECKPOINTS["robocasa"],
        "xiaomi-mibot": XIAOMI_MIBOT_CHECKPOINT,
    }[family]


def policy_binding(family):
    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.pi05_binding import make_pi05_binding
    from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding

    return {
        "pi05": make_pi05_binding,
        "smolvla": make_robocasa_binding,
        "xiaomi-mibot": make_xiaomi_policy_binding,
    }[family]()


def resolve_policy_options(args):
    """Resolve missing defaults once; retain explicit action-chunk overrides."""
    family = resolve_policy_family(args)
    args.vla = family
    checkpoint = policy_checkpoint(family)
    submitted = getattr(args, "policy_path", None)
    if submitted is not None and submitted != checkpoint.repo_id:
        raise ValueError("--policy.path conflicts with the selected VLA")
    args.policy_path = checkpoint.repo_id
    if getattr(args, "actions_per_policy_request", None) is None:
        args.actions_per_policy_request = (
            5 if family == "pi05" else
            XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request
            if family == "xiaomi-mibot" else 25
        )
    if family == "xiaomi-mibot" and not (
        1 <= args.actions_per_policy_request <= XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size
    ):
        raise ValueError(
            f"Xiaomi predicts at most {XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size} actions per request"
        )
    if family == "pi05" and args.actions_per_policy_request > 50:
        raise ValueError("Pi0.5 predicts at most 50 actions per request")
    if getattr(args, "policy_dtype", None) is None:
        args.policy_dtype = (
            XIAOMI_MIBOT_POLICY_SPEC.default_dtype if family == "xiaomi-mibot"
            else "bfloat16" if family == "pi05" else "float32"
        )
    if family == "pi05" and args.policy_dtype != "bfloat16":
        raise ValueError("the JAX Pi0.5 checkpoint requires --policy-dtype bfloat16")
    return family
