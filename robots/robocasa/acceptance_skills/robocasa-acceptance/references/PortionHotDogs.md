# PortionHotDogs: direct plate contact

Scope: the required food items assigned to each plate in `PortionHotDogs` only.

The pinned success predicate uses `check_obj_in_receptacle` for each food item.
It requires direct contact with the destination plate as well as a horizontal
center-distance bound. Being supported only by another food item on that plate
does not establish this contact. In particular, a sausage resting entirely on a
bun may look like a reasonable plated hot dog while failing the benchmark.

For placement, prefer exposed plate surface beside the other item. For acceptance,
inspect the item's actual support after release from a useful public view; a
footprint inside the rim or indirect support by food is insufficient. If all local
allocations passed but official success remains false, recheck support before
assuming that centering or additional hand withdrawal will fix it.

This note explains the plate-contact condition only. It adds no fixed placement
coordinates or new robot-control procedure and does not claim live contact truth.
