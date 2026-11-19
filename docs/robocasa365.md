# Preserved RoboCasa365 / Xiaomi protocol

This branch retains the historical 2.0 execution path for the fixed Xiaomi policy. The source is original internal version **1.2.3.5** at `c8911c64f24393e8665bd9ecc34f819173d9c3ba`. Source curation removes unrelated integrations and tooling while preserving retained runtime/model-input bytes and the original selected200 records.

## Runtime identity

- RoboCasa: version `1.0.1`, commit `456174f62b89b8fca99eaaf33949c29fec9cfc2a`.
- Robosuite: commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.
- Xiaomi checkpoint: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, revision `0d1aa76d0d82debc9b611e4d1e231096434d5be4`.
- Binding: `robocasa-xiaomi-mibot-v1`; preset: `robocasa365-pandaomron-xiaomi-mibot-v1`.
- Planning profile: `robocasa-horizon-gated-v1`.

The policy uses its original processor, observation history, crop, action decoding and instruction handling. Weights, policy specifications and capability references remain unchanged. The [installation guide](policy-setup.md) gives the supported pinned environment path.

## Planning and observations

The original horizon-gated profile retains the plan/verify/replan controller and native-success gate. Executed control actions consume the original task horizon. Source curation adds no reset, retry, horizon change or new attempt selection.

The planner uses the left, right and wrist camera interface. The retained `back_project` tool uses RGB-D grounding beyond the base policy's RGB observations, which is disclosed as part of the observation scope. While the task is in progress, tool responses can expose `environment.success=false` to the planner. The first native success stops physical execution and sets the episode score. The planner can then receive the terminal status for verification and `finish` bookkeeping.

The historical Planner configuration is `codexgateway:gpt-6-astra` with gateway-side `xhigh` and no client output-token cap (`max_tokens=0`). The 330-second HTTP timeout is distinct from the 180-second nested-planning timeout and an unset outer solve timeout. The ten selected 600-second nested-planning overrides remain historical evidence only. See the [Planner guide](../evaluations/robocasa365/2.0/PLANNER.md).

## Workload and evidence

The existing selected200 r002 panel remains **141/200 successes**, using fresh50, followup01, followup02 and followup04; followup03 stays excluded. Exact records and split denominators are in the [result report](../results/2.0-selected200/REPORT.md). Missing original state/RNG archives prevent an independent historical replay claim.

The [evaluation guide](../evaluations/robocasa365/2.0/README.md) provides a separate future reviewer profile and default-plan-only entry point. Preparing that configuration does not execute it or replace historical results. Any future native evaluation requires explicit scope and authorization.
