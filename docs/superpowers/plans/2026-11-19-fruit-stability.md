# Fruit stability and recovery

The user selected upright pears and explicitly authorized a visible planar pear
bottom, with the same apple change only if physical tests demonstrate a need.
Bananas retain their existing natural horizontal pose.

## Implementation and acceptance

1. Reproduce the saved incident offline: a pear slips during lift, then the
   latest side-lying observation cannot supersede an old upright detection.
   Keep saved dialogue and original relocation obligations private.
2. In `sim/scenarios.py`, add a same-color integral planar base to the pear.
   Renderer, RGB-D teaching samples and physical contacts use the same geometry.
   Keep total mass unchanged and let ordinary geometry determine inertia.
   Test upright settling, small tilt and low release with actual MuJoCo.
   Test apple before deciding whether its model needs modification.
3. In `assistant/catalog_perception.py`, recognize complete side-lying pears
   using measured point-cloud dimensions and synthetic side-pose examples.
   In observation merging, supersede old instances only with new camera
   evidence; retain refusal for cropped/ambiguous observations.
4. Replay the physical grip failure and isolate grasp-height, force and
   orientation effects. Change only the measured failing mechanism. Check
   actual contact lift and release, upright settled pear, natural flat banana,
   and the existing apple stacking/drop regressions.
5. Audit banana-on-yellow rejection against the saved camera neighborhood.
   Preserve real obstruction checks and required temporary-object restoration.
6. Commit distinct geometry and perception/execution changes, push, preserve
   current scene/dialogue and explicitly migrate or mark the incomplete pear
   restoration obligation. Restart the idle service and verify HTTP state,
   camera frames and physical model configuration without paid model calls.

No welds, hidden props, prescribed successful object poses, or simulator poses
used as perception detections. A fruit that falls remains a failed outcome.
