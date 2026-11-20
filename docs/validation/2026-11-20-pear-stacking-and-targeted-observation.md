# Pear stacking and targeted observation

## Reproduced failure

The saved task requested `pear` **on the green block**, not in the green area.
Before any grasp, the planner rejected all centered releases with
`no_clear_supported_release`. The green block was near the workspace boundary.
The nominal 85 mm round pear was represented by a rotated 85 mm square, including
its later leaned projection. Stacking also required the exact support center.
After those restrictions were removed in an offline diagnostic, the existing
grasp, release IK and upright-posture limits passed without a deeper grasp.

The task then repeatedly observed an unchanged scene. A planning refusal was
also incorrectly treated as a reset-required fault, preventing scene editing.
Scan counters in the incident were cumulative; they are not per-task timings.

The first live replay exposed an additional execution bug. A side inspection
physically moved the held pear away from the pre-place pose. If replanning kept
the same destination, `_inspect_release` compared only the old and new plans
and skipped the return to pre-place. The release then descended diagonally from
the inspection pose. Logs show loss of support during opening at simulation
time 1031.858; final camera motion began later, so the verification camera did
not knock over this pear. The successful earlier offline replay changed the
release plan and therefore took the return-to-pre-place branch. This is why a
full live replay was necessary in addition to the earlier passing clone.

## Changes and retained checks

- Use one conservative outline and extruded body envelope for planning and
  verification. Apples and camera-confirmed upright pears use a circumscribed
  32-sided disk; side/unknown pears and other shapes retain their rectangle.
  Nominal minimum dimensions, margins and the full leaned-body check remain.
- Search nearby centers on the measured support, preserving its interior margin.
  A rejected stack pose excludes that position/orientation, rather than an entire
  12 mm neighborhood. Search remains bounded. Every candidate still passes full
  body, finger, palm, workspace and reachability checks.
- Restore catalogue-confirmed square blocks using their equivalent quarter-turn
  orientations. The final restoration check already recognizes this symmetry;
  planning previously tried only one angle and could reject a feasible return.
  Unknown and asymmetric objects retain their requested orientation.
- After a release-inspection callback, check fresh actual TCP feedback instead
  of comparing old/new plans. Return to the pre-place pose when position or
  orientation changed, check the held-scene guard, and confirm actual arrival
  before descending. Missing/stale feedback or a falsely successful motion
  retains the object and fails safely. Stationary inspections avoid an extra move.
- Reuse complete, unconsumed grasp observations only while their capture is
  fresh, their camera pose is unchanged and scene revision/identity checks pass.
  Placement-verification images remain a separate purpose. A needed fresh image
  can be taken from the current covering view without returning to an older view.
- Verify the full goal during a local observation and stop when the visual
  postcondition is established. Stacking requires evidence of the named support;
  seeing the upper object alone does not end verification. Planning refusal without
  execution no longer triggers a settling delay.
- When two normal local views cannot see a stack's support, try one additional
  local view derived from the measured upper body and support. Its empty-gripper
  joint route is checked before moving: workspace limits, joint samples at most
  one degree apart, and conservatively expanded tool sweeps against measured
  obstacle envelopes. Missing geometry or a blocked route refuses the view.
  The temporary camera slot is reused rather than growing future scans.
- Preserve distinct measured height layers when merging observations. Prefer
  fresh support evidence over a denser view that hides the support, while retaining
  completeness and freshness priority. No simulator identity supplies a detection.
- Reuse a successful action's visual proof for subsequent `verify` and `finish`
  while the scene is unchanged. Recheck real support/contact/stability cheaply on
  every stack-proof reuse and at final publication. A physical test demonstrates
  that a 4 mm support separation can invalidate contact without crossing the
  existing 6 mm scene-revision threshold; it must still revoke success.
- Keep a healthy, empty-gripper scene editable after a geometric refusal or
  planning stagnation. Actual faults remain latched and cannot be erased by a
  later model error or approval response. `ask` authorizes an action; it is not
  evidence that a human changed the scene.

No friction, fruit shape, release-height allowance, IK tolerance, success
tolerance or physical contact criterion was relaxed. The previous 4 mm low
release allowance and actual-TCP/stationarity gate remain in force.

## Validation evidence

Private checkpoints initialize offline bodies/joints only. All subsequent
planning uses new RGB-D measurements. Offline scripts deny network access;
they do not call a paid model API. Private snapshots, conversations and scripts
stay under the runtime checkout's `.git/pear-stack-scan-20261120/`.

Two initial physical replays completed grasp, transport and stable release on
the green block. One raw-runner replay had approximately 0.44 degrees of pear
tilt, and stayed upright after another two simulated seconds. Its standard camera
views could not see the block's top surface, so the system correctly withheld
visual confirmation despite successful physical contact. This is a visibility
failure, not evidence that the pear was placed in the wrong region.

A subsequent complete dashboard-checkpoint replay through `SkillExecutor.move`,
`verify` and `finish` passed both camera and physical verification. The camera
measured a 50.7 mm green support surface from 185 visible pixels and a 111.4 mm
upright pear. Physical tilt was approximately 0.70 degrees. No held object
remained. Scan counters changed from 82 to 91 frames and 31 to 33 local scans;
full scans stayed at 2. Both the explicit verification and final completion
reused the successful action proof, with fresh physical audits.

The saved visually occluded post-release state was also replayed independently.
Two standard views still lacked support evidence; the additional guarded local
view measured the green top and confirmed the upright pear. The complete
observation used 3 frames, 1 local scan and no full scan. Maximum physical object
displacement was approximately 0.023 mm, and real support remained stable with
approximately 0.44 degrees of pear tilt. The support detector's original pixel,
surface-area and horizontal-surface requirements were retained.

An independently run final core group passed 176 tests covering stack geometry,
release replanning, pear posture, physical proof invalidation, orchestration,
stop/recovery, scene editing and layered observation merging. A separate actual
RGB-D/observation/merging group passed 50 tests. These groups overlap. The complete
pick, temporary placement, re-pick and original-position restoration regression
also passed with the production code and unchanged IK criteria.
The final scan/verification/policy/active-observation group passed 76 tests,
including the guarded fallback and interruption, nominal-size, table-clearance
and temporary-view reuse regressions. The occluded post-release replay passed
again after these last path checks were added.

After fixing the inspection return, the original dashboard scenario was replayed
specifically through the unchanged-plan branch. The callback reported
`original_changed=false`, with the actual TCP approximately 163 mm away from
pre-place. The added transit returned to `[0.1329, 0.3429, 0.201]`, followed by
the release descent. The pear stayed upright through opening and final
verification (approximately 0.68 degrees), with real green-block contact and
camera confirmation. A fresh independent skill/release-feedback/stop group
passed 73 tests; additional physical low-release and inspection tests passed.

The corrected code was then deployed and the original single-goal task replayed
through the live dashboard's production skill executor, without another model
request. The pear was placed upright on the green block: camera verification,
real support contact and stability checks all passed, with approximately 0.70
degrees of tilt. The final state was complete, with an empty gripper, no reset
requirement and scene editing available. A subsequent read confirmed that the
pear still rested on the green block at approximately 0.69 degrees of tilt.
The page and camera endpoints both returned HTTP 200. All 52 conversation
messages and the prior execution history were preserved, and model requests
remained at 40. The user's simulation speed remained at 4x.

During this successful live execution, frames increased from 93 to 102 and local
scans from 33 to 35; full scans stayed at 2. The explicit verification and finish
steps reused the successful action's evidence. The restored checkpoint did not
include its original point clouds, so this replay still required fresh initial
observations. These counts demonstrate elimination of repeated final/full-scene
scans for this run, not elimination of necessary observations after a restart,
occlusion or actual scene change.

The existing six-view pear regression contained two obsolete expectations from
the earlier pear mesh: a mandatory duplicate-pruning diagnostic and a biased
center at x=0.197. A clean archive of the pre-change HEAD reproduced its failure.
The regression now requires one complete pear, shared UI/executor detections,
and localization within the same 3 mm tolerance of the actual fixture center
(x=0.2115300883). Dedicated tests still cover duplicate-pruning diagnostics.

Validation covers the saved layout and the listed regressions. It does not imply
that all possible stacks or fully occluded supports can be recognized, nor that
an observation can be skipped after genuine scene changes.
