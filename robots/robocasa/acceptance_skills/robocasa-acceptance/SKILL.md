---
name: robocasa-acceptance
description: Consult verified task-specific plate-contact and upright-placement conditions during RoboCasa placement verification and completion review.
metadata:
  version: "0.1.0"
---

# RoboCasa acceptance reference

This is static benchmark knowledge, not a sensor reading or evidence that a
condition is currently satisfied. The runtime appends only the notes matching
the exact task name. An unlisted task has no additional criterion in this skill.

Keep the original task language unchanged. Distinguish a reasonable interpretation
of that language from an implementation-specific acceptance condition; do not
claim the original instruction explicitly stated a condition that it omitted.

Before passing a relevant placement or reviewing completed work, compare the
applicable note with fresh public images and existing tool feedback. Explain the
observed support or posture; leave occluded or ambiguous relationships uncertain.
Do not infer object contact from a reached end-effector pose or from image overlap.
The numeric tolerances in a note describe the benchmark, not an angle measured
from an image. This reference cannot approve a grasp, execute an action, reset a
budget, supply hidden state, or authorize final success. Existing Harness and
environment success authority still applies.

Only two condition families are covered in version 0.1.0:

- Plate contact: [PortionHotDogs](references/PortionHotDogs.md).
- Upright posture: [WeighIngredients](references/WeighIngredients.md),
  [RestockSinkSupplies](references/RestockSinkSupplies.md),
  [OrganizeMeasuringCups](references/OrganizeMeasuringCups.md),
  [StrainerSetup](references/StrainerSetup.md).

Do not generalize these conditions to other tasks or objects. The source revision,
task predicate files and their hashes are recorded in `manifest.json`.
