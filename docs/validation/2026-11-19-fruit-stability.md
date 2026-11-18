# Fruit stability and incident recovery

## Failure mechanism

The captured task was banana-on-yellow, with an approved temporary pear move
and a pending promise to restore the pear. The first banana release search
refused the nearby pear. During clearance, the pear was initially held but
slipped during lift. The fallen pear then failed upright-only completeness;
an older upright measurement could remain at its previous position.

Independent MuJoCo experiments reproduced three distinct mechanisms:

- The original spherical bottom supplied point support under a raised neck.
  A 3 degree initial lean settled near 114 degrees instead of upright.
- With the same RGB-D measurements and motion, the mixed task's inherited
  1.5 N foam-block grip lost the 100 g pear; the existing 2.5 N fruit force
  retained actual bilateral finger contacts. Lift feedback showed a smaller
  seated opening than the initial close feedback.
- An oblique visible-surface rectangle biased fruit centers toward the camera.
  Treating a nominal circular belly as a rotating square added fictitious
  corners at placement verification. Both camera and physical audit paths
  needed appropriate geometry; changing only a success tolerance was rejected.

## Resulting behavior

The pear has a visible integral 48 mm planar base, shared by rendering,
RGB-D samples and collision geometry. Mass remains 100 g. Apples passed the
control experiments without a geometry change. Bananas retain their natural
horizontal model and resting semantics.

Complete side pears are distinguished using measured 3-D extents and axis
orientation, with side-pose synthetic teaching samples. Cropped bulb-only
views remain invalid. A fresh complete singleton replaces an old location;
old upright height no longer blocks reacquisition. Complete upright round
fruit use a checked measured top cap to reduce visible-surface center bias.

The mixed task uses 2.5 N and updates held width after lifting. An observed
upright pear's predicted release tilt is limited to 15 degrees; outcome checks
still require actual support, low motion and upright posture. Zone audits use
the actual body's directional support function and require its whole extent
inside the zone. Completion skips, explicit verify and final results share
these audits. Failures preserve their original goals and restoration debt.

If a MuJoCo inspection move times out or stalls, the driver first holds its
measured joints. With fresh pose, health and held feedback, the old route ends
and the actual camera view can be measured before replanning. A failed motion
is never relabeled successful; fault, abort and drop still propagate.

## Verification

Offline verification blocks online connections. Fresh checks include:

- 33 physical resting, release-posture, inspection-feedback and catalog checks.
- 39 RGB-D/classification/observation checks after center correction, including
  real camera images of complete side pears and invalid partial views.
- 74 completion, batch, release-posture and inspection checks, including the
  former stale test-double signature in free placement.
- 129 physics, skill protocol, task skill and active observation checks.
- 415 JavaScript assertions across the five UI suites.
- Final real-contact integration: 3 passed in 147.23 s, covering task force
  configuration, camera-measured pear lift and upright release to a specified
  tabletop point and green zone, plus explicit green-zone verification.

The broader execution run passed 54 cases and exposed one old test-double
signature missing `held_only`; after forwarding that argument, the specific
scenario and the 74-case group passed. Old legacy scenario tests still contain
unrelated blue-block/ROI assumptions and are not claimed clean.

## Operational limits

A complete side-pear detection is not an automatic righting skill. Current
grasps change horizontal approach/yaw and cannot reliably stand an arbitrary
fallen pear up. The model and UI disclose this and request an explicit human
scene edit. The saved incident must retain its side pose and pending pear
restoration across deployment; rebuilding a model is not a robot success.

Strict point placements retain their 15 mm tolerance. Real contact slip may
still require a new observation and corrective grasp. No welded attachment,
hidden fixture, prescribed successful object pose, or ground-truth detection
is used to make these checks pass. No browser visual inspection is claimed;
UI rendering/listeners, HTTP state and camera endpoints are checked separately.
