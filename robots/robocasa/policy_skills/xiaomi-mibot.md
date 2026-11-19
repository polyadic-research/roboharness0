# Xiaomi-Robotics-1 capability guidance

Policy: xiaomi-mibot; checkpoint: XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365;
HF revision: 0d1aa76d0d82debc9b611e4d1e231096434d5be4.
Measurement dates: 2026-10-12–2026-10-13; measured sources: xiaomi-capability-20261012 and xiaomi-capability-round2-20261012; 65 atomic tasks × 15 episodes and
5 composite tasks × 5 episodes, RoboCasa 1.0.1 pretrain, long horizon, environment
seed 7 and key-derived independent policy seeds. Predictions and executed chunks
both contain 16 control steps. Typical steps use successful first-success medians.
Scenes use native original-round RNG restoration, then saved-state recovery
on state mismatch. If saved recovery fails while the native reset is valid,
that unchanged native scene is scored unpaired; its state hashes and restore
error are recorded. Unmatched samples remain in rates and timing and are
excluded from paired comparisons.
The separate fixed D01–D10 studies (25/30 and seed-0 run05 9/10) are background
and do not contribute to these rows. Two-per-instruction D07 probes show subgoal
responses on that fixed scene; controllability across scenes remains unknown.
Evidence and pairing differences: reviews/xiaomi-pure-eval-20261011/report.md and reviews/xiaomi-capability-round2-20261012/report.md.

Read this skill before every plan, replan, intervention decision or instruction
change. Existing tool constraints, mandatory reviews, verification, recovery
and success authority remain binding. The machine-readable
`xiaomi-mibot.capabilities.json` is the authority for tiers and declared entries; the
table below is its readable form, and registered plans are checked against
the file.

## 任务范围判定 / Task scope

- **Short**: a single operation matching one table row or a declared entry,
  regardless of tier. Pick-and-place is one complete operation. For delegate
  and monitor tiers deliver the original task_language verbatim as one
  complete operation; added observation/verification never fragments it into
  approach/grasp/lift steps. A weak short task is one takeover step (Control
  modes below); its monitor fallback uses the original sentence.
- **Short, none**: one operation with no matching row or declared entry; tier
  none, control_mode=takeover (monitor when the TAKEOVER section is absent).
- **Unmeasured composite, delivered whole**: start with one whole-task `monitor`
  step (`skill_row=none`, `tier=none`), complete original instruction and
  `intervene=false`. Preserve the native policy stream. Observation and grounded
  semantic replanning are distinct from motor takeover: a fresh current-goal
  phase boundary, first explicit physical miss, or sustained current-goal stall
  may open policy-only monitor replanning. Preserve operation/subtask identity
  and counters. Internal phase changes do not require changing the VLA sentence.
  Direct motor recovery is only permitted after two explicit physical failures,
  or one destructive failure/confirmed target conflict, and is never compulsory.
- **Long**: two or more ordered operations, including multiple targets/containers
  or placement followed by appliance control, unless delivered whole (next item).
  Count operations in task content, not verbs or sentence length. Use the catalogs
  for reference; unlisted names follow these content criteria.
- **Long, delivered whole**: this policy performs some long tasks end to end. When
  the live task's own composite row (the current task's row, marked family
  composite in the planning request) is reliable or unstable, i.e. its measured
  pure-policy success reaches the unstable lower bound **30%**, classify the task
  as short: one complete operation whose skill_row is that row, with its tier
  and the mode by tier (reliable delegate, unstable monitor), the original
  task_language verbatim as the instruction, and expected_evidence covering every
  requested outcome. Matching the task to its own row is not matching a composite
  row to one of its operations. Do not decompose it up front. A grounded failure
  inside the step (a missed grasp, a drop, a stall) is retried in the same step
  with the same sentence and intervene=false: the policy recovers within the
  task on its own. An entry under control_mode_escalations permits decomposition
  and does not require it; switch to the long-task flow only when fresh evidence
  shows the policy cannot complete the remaining outcomes, keeping outcomes
  already verified. Such an escalation is answered this way, not by a staged
  takeover of the composite row. A measured weak own row leaves the task
  long from the start; an absent own row follows the unmeasured-composite rule above.
  The threshold makes the task eligible; it does not
  guarantee success.

**65 short reference tasks**: the 65 task names in the operation table below.
**5 measured long reference tasks**: CategorizeCondiments, PanTransfer, SteamInMicrowave, StoreLeftoversInBowl, WashLettuce. [M]

## Control modes

Mode by tier: reliable → delegate; unstable → monitor; weak or none →
takeover; the carry between two operations at different fixtures in a long
task → transition. Name every operation once (operation) and keep the name, with
its subtask names (pick, carry, place, grasp, pour), across replanning: the
harness counts steps, physical failures, progress and LLM quotas per operation
and subtask and lists fallbacks and escalations in the planning request. [S]

## Decisions

Reliable (≥70%): delegate the whole operation with the original task_language
or the operation instruction selected once under Instruction format below.
Keep its instruction and skill-level step
while progress continues. Without clear failure evidence, do not subdivide it
into approach/close/lift/waypoints or add discretionary arm corrections. [S]
Unstable 30–<70%: try first; monitor target identity, motion, source support,
measured gripper width and EEF progress together. A grounded miss, drop,
collision or sustained stall warrants physical intervention; occlusion/call limits alone do not.
For delegate and monitor steps, sustained stall means past the matched row's typical steps and no task-directed
progress, assessed across all axes using the target and measured robot motion.
Contact constraining one axis while the hand or target advances in the task
direction is productive work. Before typical steps, do not make stall-driven
physical corrections without evidence of a drop, collision or wrong target;
productive target contact is not a collision exception. When uncertain,
continue one configured default policy window and observe again.
Typical steps are not a deterministic completion deadline, but they set the
escalation point: a whole operation on an exact row that has run 1.5× its
typical steps and then 120 more steps without progress, or any delegate or
monitor subtask after two verified explicit physical failures may be replanned as takeover
(CONTROL MODES). A long task delivered whole, split subtasks, declared entries and none-tier operations
have no step threshold; composite current-goal stalls request semantic policy review, not automatic motor control. A slow policy that keeps progressing is left alone. The same progress and
timing criteria bind adjust_eef, move_base and adjust_torso; none of this
timing protection binds takeover or transition steps.
Weak <30% or none: control_mode=takeover. Drive the complete operation under
TAKEOVER from public evidence with explicit completion checks; match
source/destination and actuator exactly or use tier none; the
policy receives no instruction until the fallback hands the operation back
with its frozen sentence. A takeover runs on its step and call quota, not on a
failure count: retry with a changed approach while the quota lasts. When the
harness ends it (control_mode_fallback), replan the subtask as monitor; if the
policy then fails or stalls, the harness offers takeover again. Long tasks
follow Task scope above and the generic
decomposition flow. [S]

## Operation boundaries

Typical steps = success-only median(first_success_step), in control transitions;
not expected duration or a guarantee. Use generic allocation guidance; tiny
success counts are especially uncertain. * marks Wilson 95% crossing a tier
boundary. Point tiers are provisional; pooled categories never override routes.
Failure modes are unaudited unless cited; examples are not frequencies. [S]

Each row quotes its official task instruction. An operation matches a row when
its source, destination and actuator fit that wording, not a shorter label: a
generic object or container in it covers any instance of that kind, so "the
container located on the counter" includes a bowl or plate on the counter. [S]

| Task (official instruction) | Tier | 典型步数 / Typical steps | Attention |
|---|---|---:|---|
| CloseBlenderLid: "Close the lid blender by securely placing the lid on top." | Unstable* (9/15) | 361 | 未知 |
| CloseFridge: "Close the fridge door(s)." | Reliable* (12/15) | 264 | 未知 |
| CloseToasterOvenDoor: "Close the toaster oven door." | Reliable* (13/15) | 169 | 未知 |
| CoffeeSetupMug: "Pick the mug from the counter and place it under the coffee machine dispenser." | Weak* (3/15) | 199 | 未知 |
| NavigateKitchen: "Navigate to the {kitchen location}." | Reliable* (13/15) | 159 | 未知 |
| OpenCabinet: "Open the cabinet door(s)." | Reliable* (11/15) | 382 | 未知 |
| OpenDrawer: "Open the {left/right} drawer." | Reliable* (13/15) | 179 | 未知 |
| OpenStandMixerHead: "Open the stand mixer head." | Reliable (14/15) | 93 | 未知 |
| PickPlaceCounterToCabinet: "Pick the item from the counter and place it in the cabinet." | Reliable* (11/15) | 202 | 未知 |
| PickPlaceCounterToStove: "Pick the item from the plate and place it in the pan." | Reliable (14/15) | 199.5 | 未知 |
| PickPlaceDrawerToCounter: "Pick the item from the drawer and place it on the counter." | Reliable (15/15) | 314 | 未知 |
| PickPlaceSinkToCounter: "Pick the item from the sink and place it on the container located on the counter." | Reliable (14/15) | 237.5 | 未知 |
| PickPlaceToasterToCounter: "Place the toasted item on a plate." | Reliable* (13/15) | 231 | 未知 |
| SlideDishwasherRack: "Fully slide the top dishwasher rack {in/out}." | Reliable (14/15) | 168.5 | 未知 |
| TurnOffStove: "Turn off the [burner location] burner of the stove." | Unstable* (9/15) | 267 | 未知 |
| TurnOnElectricKettle: "Press down the lever to turn on the electric kettle." | Reliable (14/15) | 92 | 未知 |
| TurnOnMicrowave: "Press the start button on the microwave." | Unstable* (8/15) | 110 | 未知 |
| TurnOnSinkFaucet: "Turn on the sink faucet." | Reliable (14/15) | 203 | 未知 |
| AdjustToasterOvenTemperature: "{Increase/Decrease} the toaster oven temperature." | Unstable* (5/15) | 200 | Verify the result; failure modes unaudited. [D] |
| AdjustWaterTemperature: "The water running in the sink is {hot/cold}. Adjust the faucet handle to run the water {cold/hot}. Make sure to keep the water on." | Reliable* (13/15) | 192 | Keep full operation unless clear failure. [D] |
| CheesyBread: "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish." | Reliable (15/15) | 302 | Keep full operation unless clear failure. [D] |
| CloseCabinet: "Close the cabinet door(s)." | Reliable* (13/15) | 300 | Keep full operation unless clear failure. [D] |
| CloseDishwasher: "Close the dishwasher door." | Reliable (14/15) | 122.5 | Keep full operation unless clear failure. [D] |
| CloseDrawer: "Close the {left/right} drawer." | Reliable (15/15) | 118 | Keep full operation unless clear failure. [D] |
| CloseElectricKettleLid: "Close the lid of the electric kettle." | Reliable (15/15) | 60 | Keep full operation unless clear failure. [D] |
| CloseFridgeDrawer: "Fully close the fridge drawer." | Unstable* (10/15) | 105.5 | Verify the result; failure modes unaudited. [D] |
| CloseMicrowave: "Close the microwave door." | Reliable (15/15) | 174 | Keep full operation unless clear failure. [D] |
| CloseOven: "Close the oven door." | Reliable* (11/15) | 145 | Keep full operation unless clear failure. [D] |
| CloseStandMixerHead: "Close the stand mixer head." | Reliable (14/15) | 75.5 | Keep full operation unless clear failure. [D] |
| CoffeeServeMug: "Pick the mug from under the coffee machine dispenser and place it on the counter." | Reliable* (12/15) | 169 | Keep full operation unless clear failure. [D] |
| LowerHeat: "Lower the heat of the kettle." | Unstable* (8/15) | 295.5 | Verify the result; failure modes unaudited. [D] |
| MakeIcedCoffee: "Pick up an ice cube and place it in the glass of coffee." | Reliable* (13/15) | 238 | Keep full operation unless clear failure. [D] |
| OpenBlenderLid: "Open the blender by taking off the lid and placing it on the counter." | Reliable* (11/15) | 162 | Keep full operation unless clear failure. [D] |
| OpenDishwasher: "Open the dishwasher door." | Unstable* (9/15) | 148 | Verify the result; failure modes unaudited. [D] |
| OpenElectricKettleLid: "Press the button to open the lid of the electric kettle." | Weak* (4/15) | 56 | Takeover; verify the result. [D] |
| OpenFridge: "Open the fridge door(s)." | Reliable* (13/15) | 382 | Keep full operation unless clear failure. [D] |
| OpenFridgeDrawer: "Fully open the fridge drawer." | Unstable* (10/15) | 163 | Verify the result; failure modes unaudited. [D] |
| OpenMicrowave: "Open the microwave door." | Reliable (15/15) | 227 | Keep full operation unless clear failure. [D] |
| OpenOven: "Open the oven door." | Reliable (15/15) | 122 | Keep full operation unless clear failure. [D] |
| OpenToasterOvenDoor: "Open the toaster oven door." | Reliable* (13/15) | 145 | Keep full operation unless clear failure. [D] |
| PackDessert: "Add the [*dessert*] to the tupperware that contains the [*food*]." | Reliable (14/15) | 228.5 | Keep full operation unless clear failure. [D] |
| PickPlaceCabinetToCounter: "Pick the item from the cabinet and place it on the counter." | Reliable* (11/15) | 191 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToBlender: "Pick the item from the counter and place it in the blender." | Reliable (15/15) | 339 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToDrawer: "Pick the item from the counter and place it in the drawer." | Reliable (14/15) | 227 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToMicrowave: "Pick the item from the counter and place it in the microwave." | Unstable* (10/15) | 380.5 | Verify the result; failure modes unaudited. [D] |
| PickPlaceCounterToOven: "Place the item on the {bottom/top} rack of the oven." | Reliable* (13/15) | 258 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToSink: "Pick the item from the counter and place it in the sink." | Reliable* (12/15) | 182 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToStandMixer: "Place the item in the stand mixer bowl." | Reliable* (12/15) | 217 | Keep full operation unless clear failure. [D] |
| PickPlaceCounterToToasterOven: "Place the item on the rack of the toaster oven." | Reliable (14/15) | 199.5 | Keep full operation unless clear failure. [D] |
| PickPlaceFridgeDrawerToShelf: "Pick the item from the fridge drawer and place it on a fridge shelf." | Reliable (14/15) | 206.5 | Keep full operation unless clear failure. [D] |
| PickPlaceFridgeShelfToDrawer: "Pick the item from the fridge shelf and place it in the fridge drawer." | Reliable (15/15) | 219 | Keep full operation unless clear failure. [D] |
| PickPlaceMicrowaveToCounter: "Pick the item from the microwave and place it on container located on the counter." | Reliable* (13/15) | 270 | Keep full operation unless clear failure. [D] |
| PickPlaceStoveToCounter: "Pick the item from the pan and place it in the container." | Unstable* (10/15) | 185 | Verify the result; failure modes unaudited. [D] |
| PickPlaceToasterOvenToCounter: "Pick the item from the toaster oven and place it on the plate on the counter." | Unstable* (10/15) | 160.5 | Verify the result; failure modes unaudited. [D] |
| PreheatOven: "Preheat the oven by turning the temperature knob." | Unstable* (7/15) | 164 | Verify the result; failure modes unaudited. [D] |
| SlideOvenRack: "Fully slide the oven rack {in/out}." | Reliable (15/15) | 210 | Keep full operation unless clear failure. [D] |
| SlideToasterOvenRack: "Fully slide the toaster oven tray {in/out}." | Reliable* (13/15) | 92 | Keep full operation unless clear failure. [D] |
| StartCoffeeMachine: "Press the button on the coffee machine to serve coffee." | Weak* (3/15) | 88 | Takeover; verify the result. [D] |
| TurnOffMicrowave: "Press the stop button on the microwave." | Weak* (3/15) | 140 | Takeover; verify the result. [D] |
| TurnOffSinkFaucet: "Turn off the sink faucet." | Unstable* (10/15) | 95 | Verify the result; failure modes unaudited. [D] |
| TurnOnBlender: "Turn on the blender by pressing the power button." | Unstable* (6/15) | 86.5 | Verify the result; failure modes unaudited. [D] |
| TurnOnStove: "Turn on the [burner location] burner of the stove." | Unstable* (9/15) | 190 | Verify the result; failure modes unaudited. [D] |
| TurnOnToaster: "Push down the lever of the toaster to turn it on." | Unstable* (9/15) | 67 | Verify the result; failure modes unaudited. [D] |
| TurnOnToasterOven: "Turn on the toaster oven by setting the timer." | Unstable* (6/15) | 253.5 | Verify the result; failure modes unaudited. [D] |
| TurnSinkSpout: "Turn the sink spout to the {left/right}." | Reliable* (11/15) | 83 | Keep full operation unless clear failure. [D] |

No declared entries exist in this measured catalog. An operation without an
exact measured atomic row is tier none. Composite rows describe whole tasks and
never stand in for a different atomic operation. After loss, re-establish
same-target acquisition; a closed gripper alone cannot prove retention.

## Transitions

Only in long tasks, between operations at different fixtures when the base
must move (target beyond arm reach); the carry of such a pick-and-place is a
transition between its pick and place steps. Within reach (for example a
cabinet above the counter) the pick-and-place stays one operation.
Set travel_m from back_project; the carry's quota is 150 steps plus 200 per
metre.

Handoff poses below summarize the starts of this policy's successful atomic
samples. They remain provisional. Fixture rows require at least three successful
episodes; matched-task rows and their sample counts are in the capability JSON.
Navigation starts are excluded: they do not describe a fixture-facing operation
handoff. The source records all 13 excluded NavigateKitchen successes. Of 746
successful atomic episodes, 157 have no usable measured start pose; the remaining
576 non-navigation starts contribute to these statistics. Missing-pose reasons are recorded in the evaluation report.
Distances refer to the fixture front along its normal, not collision clearance.
Use the matched task's distance, lateral range, heading and base-relative EEF
start pose; ground the live target with public observations before handing back.

<!-- handback:begin -->
| Fixture | Base to front (m): median [p10–p90] | Fixture centre lateral (m) | Heading max | Arm start, base frame (m) | Episodes |
|---|---|---|---:|---|---:|
| Counter | 0.17 [0.12–0.32] | -0.87 to +1.07 | 0° | [0.25, -0.00, 0.60] | 190 |
| Dishwasher | 0.37 [0.24–0.41] | -0.58 to +0.56 | 0° | [0.25, -0.00, 0.59] | 37 |
| Drawer | 0.29 [0.23–0.45] | -0.63 to +0.65 | 0° | [0.25, -0.00, 0.60] | 31 |
| ElectricKettle | 0.24 [0.19–0.31] | -0.12 to +0.08 | 47.4° | [0.25, -0.01, 0.59] | 33 |
| FridgeBottomFreezer | 0.29 [0.25–0.52] | -0.32 to -0.03 | 0° | [0.25, -0.00, 0.60] | 18 |
| FridgeFrenchDoor | 0.28 [0.23–0.56] | -0.10 to +0.12 | 0° | [0.25, -0.00, 0.59] | 43 |
| FridgeSideBySide | 0.27 [0.22–0.57] | -0.13 to +0.14 | 0° | [0.25, -0.01, 0.59] | 13 |
| HingeCabinet | 0.37 [0.29–0.42] | -0.07 to +0.14 | 0° | [0.25, -0.01, 0.60] | 21 |
| Microwave | 0.29 [0.21–0.39] | -0.13 to +0.10 | 0° | [0.25, -0.01, 0.60] | 39 |
| Oven | 0.34 [0.05–0.48] | -0.54 to +0.68 | 0° | [0.25, -0.01, 0.60] | 58 |
| SingleCabinet | 0.38 [0.29–0.39] | +0.04 to +0.15 | 0° | [0.25, 0.00, 0.59] | 4 |
| Sink | 0.26 [0.20–0.28] | -0.08 to +0.13 | 0° | [0.25, -0.01, 0.60] | 13 |
| Stove | 0.16 [0.11–0.19] | -0.12 to +0.11 | 0° | [0.25, -0.00, 0.59] | 25 |
| ToasterOven | 0.36 [0.32–0.44] | -0.12 to +0.10 | 0° | [0.25, -0.01, 0.60] | 50 |
<!-- handback:end -->
Do not borrow another policy's fixture distances or arm start poses. navigate_to
requires an explicit public relative target or a back_project point with a
nonnegative standoff_m; move_eef_path requires public base-frame waypoints.
For fixtures without measured handoff poses, ground the target and clearance from fresh
observations, keep the present safe arm pose during navigation, and change it
only when required to avoid contact or reach the next operation. Controller
anchoring retains the current achieved arm pose; it supplies no learned start
pose. Observe, resolve verification and hand back with the operation's frozen
sentence. An omitted pose is not evidence that the training pose was reached.

Navigation: the base translates without turning first. In a narrow space
(between stools or chairs, beside an island) back straight out first, move in
segments of at most about 1 m with an observe after each, and turn only in
open floor. While carrying, keep the gripper closed and the object clear of
surfaces; let the tools rate-limit.

## Takeover

Takeover applies to weak rows and to none-tier operations (pour or tilt,
cookware back onto an appliance, any source→destination pair without a row),
and to any operation the harness lists under control_mode_escalations.
Default is full takeover including contact for weak rows and none-tier
operations. An escalated takeover of a reliable or unstable row is staged: the
LLM brings the base and hand to the handoff pose (Transitions above), then the
policy makes the contact with its frozen sentence (LLM-driven contact is not
validated yet). No weak or none operation type is marked "staged takeover"
yet; the fork experiments decide per operation type and the marking goes into
the row's Attention column. The monitor fallback after the harness ends a
takeover uses the operation's frozen sentence.

Split takeover (policy grasps, LLM continues; provisional until the fork
experiments): pouring or dumping the contents of held cookware (a pan or pot
onto a plate or into a container). Plan grasp (monitor, smolvla_pick with the
operation's frozen sentence; it stops at the confirmed grasp), pour (takeover:
pour_into with the receiver's back_projected centre as its target; the tool
lifts the pour rim to container depth plus 4 cm above the receiver, moves it
over the target, rotates the cookware about that rim in 10 degree steps up to
90 degrees, dwells and returns upright; it locates the cookware from the hand
pose, so pass only the target; it stops on grasp loss; call it with its default
call length and, if the segment ends with the cookware tilted, right it with
pour_into and return_upright rather than wrist rotations) and place (monitor with
the frozen sentence, or takeover with place_container at the freshly
back_projected destination centre when the policy does not set the held
cookware down; the policy may have moved the cookware while lifting it, so the
hand pose at the handover is not its original place). In the π0.5 2026-10-11 pan-transfer fork probe this rim pivot landed
the vegetable on the plate and the task succeeded, while the same tilt about
the hand dropped it on the counter; verify the contents from fresh images of
the receiver, not from the pour result. With this policy, a harness run of the
pan-transfer task grasped the pan and stopped at the confirmed grasp, pour_into
landed the vegetable on the plate (the tilt stopped at 88 degrees on suspected
contact and return_upright righted the pan) and the task succeeded; one run.

## Instruction format

Short tasks use the live original task_language verbatim, for example “Open the
stand mixer head.” For a long-task operation, quote the corresponding original
task sentence verbatim. If one sentence covers several separately verified
operations, use the same original sentence for every covered operation; let the
policy choose the internal order where the task has not required an order.
If no original sentence covers the operation at that granularity, write one
task-style imperative once and freeze it for that operation, including across
replanning. Ground its targets/relations in public observations; no coordinates,
finger control or scripted motor sequence. Retries are physical actions, never
paraphrases: keep the chosen instruction through stalls, failures, corrections
and recovery. Default to switching language only on entry to a new registered
operation. After an authorized semantic review, fresh current-goal evidence
may justify an explicit language intervention within the same operation;
preserve its identity, counters and original outcomes. Internal phase
bookkeeping alone never requires a language switch. In transition steps, and in takeover steps except the contact phase of a staged
takeover, the policy does not run and receives no instruction; when control
returns to it, use the sentence already frozen for that operation. [S]

Articulated pull or slide (for example the reliable dishwasher-rack row): while
the hand stays on the handle or crossbar and the measured base-relative x
advances in the pull direction with z constrained, the policy is pulling; treat
this target-contact motion as progress, not a stall of the blocked axis. Preserve productive contact, apply the generic timing protection and keep
mandatory reviews.

Evidence:
- [S] Measured Xiaomi: xiaomi-capability-20261012, 270 atomic and 25 composite episodes; VLM/reviews/xiaomi-pure-eval-20261011/report.md. Official aggregate rates are separate background evidence.
- [D] Shared Harness control-mode, verification and recovery rules; default Attention text is not a measured failure-mode frequency.
- Reproducible measured source, paired statistics and trace/pose export: VLM/reviews/xiaomi-pure-eval-20261011/.

- [M] Combined measured Xiaomi rows and opening poses: xiaomi-capability-20261012 and xiaomi-capability-round2-20261012 (65×15 atomic, 5×5 composite); see the local evaluation report and handback.json.

- [M] Supplemental 47 atomic tasks × 15 episodes: xiaomi-capability-round2-20261012; combined with the original 18 atomic tasks. Composite coverage remains the original five tasks until part B completes.
