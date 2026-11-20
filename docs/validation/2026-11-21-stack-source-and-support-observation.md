# Joint observation before stacking

## Reproduced incident

The requested task was to put the apple on the green block. Both objects had
complete measurements before each attempted move. The fresh pre-grasp image
then covered the apple but cropped the green block at the image border.
Local observation merging replaced the earlier complete green-block measurement
with the new invalid fragment. Destination validation consequently refused the
move before any grasp, with the detail that the support must have a unique valid
3-D measurement. Observing the green block restored its measurement, but the next
apple-only confirmation repeated the problem.

Both the goal reference and the direct `on_object` destination resolved correctly.
The generic `invalid_destination` message hid the measurement failure from the
model, which eventually stopped after repeated observations. This was an
observation dependency failure, not unsupported command syntax.

An isolated MuJoCo replay of the saved idle scene reproduced the exact
`invalid_destination` refusal through `SkillExecutor.move` after one fresh image,
before physical execution. Both the user interface and the replay retained an
empty gripper and did not require a reset. The replay denied network access.
Private state and conversations remain outside tracked files in the runtime
checkout's `.git/apple-stack-20261121/` directory.

## Repair criteria

Stack preparation must confirm both the grasp target and its named support from
the current observation transaction. It should prefer a view covering both and
stop as soon as the necessary complete measurements and grasp geometry exist.
Incomplete or stale support geometry must not become valid by retaining old
coordinates or relaxing validation. Interventions and cancellation must still
invalidate preparation. Normal single-object observation must retain its
existing behavior.

The model must receive the bounded textual reason for a refusal, without private
simulator state. Verification must exercise the original scene through the normal
skill executor, real contact, final visual confirmation and completion checks.

## Changes

- Both semantic moves and direct picks prepare the grasp target and named support
  together. Camera ranking prefers joint coverage; early stopping requires a
  feasible grasp and a unique complete support from the same observation batch.
- Multi-view preparation combines complementary full measurements rather than
  asking the model to alternate between the upper object and its support.
  The ordinary cache-merging rule remains unchanged, so an invalid new object
  cannot be silently replaced with its old valid coordinates.
- Reusing a source capture also requires a fresh complete support. Direct
  execution repairs missing support observations before planning, passes only
  current-batch support geometry to the planner, and refuses interrupted or
  invalidated observations before grasping.
- Existing unsupported curved support classes are rejected before camera motion.
  The allowed support types remain the existing flat-topped blocks.
- Failure history and model context retain the specific `detail` text, limited
  to 1,000 characters. Structured private data is excluded.

## Validation

The saved incident passed an isolated replay through the production move,
verify and finish paths. The apple was physically supported by `green_cube`,
with camera confirmation, stable contact, no held object and approximately
1.63 degrees of tilt. Counts changed from 118 to 122 frames and 42 to 44 local
observations; full scans stayed at 3. Both verify and finish reused the action's
proof and rechecked physical support. No paid model call was made.

The feedback, model-context and orchestration groups passed 129 tests, including
the refusal-to-observation-to-replanning chain and bounded/private-data cases.
The new joint-observation tests first failed on the upper-only early stop, then
covered complementary views, single-view completion, missing/ambiguous/stale
support, direct and semantic execution, interruption, and unsupported support
rejection without camera motion.

The previous pear-on-green saved scene also passed a complete physical replay
with camera confirmation and real stable contact. It used the same observation
counts as its previous successful baseline: 93 to 102 frames, 33 to 35 local
observations, and 2 full scans unchanged. Restoring summary-only checkpoints
still requires rebuilding missing point-cloud evidence; this can visit multiple
camera poses within a local observation and is not a claim of zero camera motion.

Related test groups passed 50 tests (joint observation and stack geometry),
67 tests (joint observation, basic skills and already-completed steps), and
56 tests (the earlier joint cases, targeted scans and observation policy).
These groups overlap and some predate the two unsupported-surface cases.
An independent review found an existing proof-test fixture without a session;
the same failure reproduced from the pre-change revision. The fixture now uses
a configured session while still proving that an edit revision alone cannot
establish stacking success. Its full protocol file passed 21 tests.

## Additional live findings

The first live replay physically placed the apple on the green block, but its
two final views could not confirm the support. The visible green top patch was
below the existing pixel, extent and area requirements. All four original
look-at fallback poses were unreachable for this central-table location; their
fixed camera offset put the requested flange behind the reachable workspace.
The system correctly withheld success even though independent contact passed.

Fallback observation now also samples a bounded set of nearby taught joint poses,
checking joint limits, complete measured image coverage and the original full
tool sweep. At most three different local fallback captures follow the two normal
views, stopping immediately when the camera confirms the goals. Reusing the
transient view slot preserves each distinct capture in the current merge.
Matching historical support geometry may aim the camera and add an obstacle;
it is explicitly invalid as a detection and never supplies success evidence.
The support detector's thresholds, IK tolerances and collision checks remain.

With tighter contact integration, an isolated replay of the first live state
failed with the old two-view fallback and passed with the added safe local view
(three captures, one local observation, no full scan). Object movement was less
than 0.028 mm. A separate newly released state required four captures before
camera and physical verification both passed, demonstrating why a single extra
view is insufficient for all nearby release poses.

The live stack also exhibited persistent numerical creep. A single-threaded
physics replay reproduced approximately 0.654 mm/s, ruling out UI or idle-thread
interference. Increasing contact-solver iterations alone or tightening its
tolerance alone was insufficient. Thirty no-slip iterations and 1e-10 tolerance
reduced a public off-center stack fixture's 20-second displacement from 14.49 mm
to approximately 0.01 mm, without changing friction, mass, gravity, body state
or success criteria. Removing the support still caused a real gravity-driven fall.

Applying that precision globally regressed the existing held-pear release
planning and increased physics computation cost. The final policy therefore
retains the configured manipulation parameters (default 5 / 1e-6), raising
precision only for an empty gripper with real stacked-body contact during
settling, observation or idle time. Starting execution, retaining a held-recovery
record or losing support restores the configured parameters. Rebuilds reset the
temporary mode, and existing higher-precision configurations are not downgraded.
This suppresses released/idle stack creep; other stacks can still accumulate
some residual drift while a manipulation uses the normal global solver settings.

The final staged solver policy passed complete physical apple and pear replays.
The apple used 118 to 122 frames and 42 to 44 local observations, with 3 full scans
unchanged; the pear used 93 to 102 frames and 33 to 35 local observations, with
2 full scans unchanged. Both ended with an empty gripper, no reset requirement,
camera confirmation and stable real support. The apple tilt was approximately
1.66 degrees. The staged policy's lifecycle and falling-support tests passed
13 cases. The additional camera and cache group passed 73 cases, with 6 deselected.

A subsequent live verification exposed a distinct old-state case: before the
precision fix, the released stack had already crept approximately 18 cm from its
last complete observation. The saved support hint remained near the old location.
Fresh camera images found the apple near the green placement region, but three
reachable optical fallback poses viewed almost the same occluded side. Changing
only the initial named view did not fix this: five captures still lacked support
evidence. The physical stack remained supported and editable; success was
correctly withheld. This finding requires a visibly different viewing direction,
not relaxed support thresholds or re-grasping an already placed object.

The follow-up camera repair samples both sides and both elevation directions
around the two most relevant taught views. Candidates must still cover the
measured target and pass joint limits and the complete tool sweep. Their minimum
viewing-direction difference from previous captures orders the candidates; it
is a soft preference, not a new acceptance threshold. After one extra optical
view, the next attempt considers these different directions before another
near-identical optical pose. The total budget remains three extra captures.
If a remembered base is too far from the current upper-body measurement for
their footprint bounds to overlap, its old position remains an obstacle but
does not constrain image coverage or view ranking. It is not relocated or
published as a valid detection.

Unmodified production verify/finish replays passed from both the pre-update
snapshot (with the old camera summary) and the latest live failure snapshot
(with the displaced upper-body measurement). Each required four local captures,
with no new full scan and no re-grasp. In the latest snapshot, frames changed
from 128 to 132 and local observations from 45 to 46; full scans stayed at 3.
Camera evidence, stable physical contact and task completion all passed, with
approximately 1.27 degrees of apple tilt. The 60 messages and 48 model requests
were preserved. Saved private results are `production-drifted-stack-success.json`
and `production-latest-stack-success.json` in the private diagnostic directory.

The final camera, targeted-observation, merge, layered-perception and goal-check
regression group passed 71 tests. It includes different viewing directions with
unchanged optical-axis rotation, a displaced remembered support retained as an
obstacle, bounded retries, cancellation and revision changes. An independent
review found no blocking production-code issue. The larger finite candidate set
can require more geometry checks; observations still stop at the first valid
support measurement rather than visiting every candidate.

## Final live verification

The running service was updated from a saved empty-gripper checkpoint, preserving
the already placed objects, dialogue, task goal and failure history. Production
verify and finish completed successfully without repeating the move. The live
result matched the latest replay: 128 to 132 frames, 45 to 46 local observations,
3 full scans unchanged, real apple-to-green contact and stable upright posture.
The service ended complete and editable, with no held object or reset requirement.
All 60 messages and 48 model requests were preserved. The page, live camera and
final observation image returned HTTP 200. The final live state and images are
saved privately, outside version control.
