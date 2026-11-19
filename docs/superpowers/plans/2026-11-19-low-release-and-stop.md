# Low release and recoverable task stop

**Goal:** Place objects close to measured support surfaces and allow interaction after a task stop without resetting the scene.

**Design:** The interactive task uses a 4 mm nominal bottom-to-support clearance for flat table and stack placements, including a depth/grip error allowance. A first 2 mm experiment made the pear contact and slide before opening (actual gap -0.6 mm); it was rejected. Opening requires fresh actual TCP position/orientation feedback at the planned release and a short stationary interval. A failed descent must never use the former 80 mm above-target allowance to release. No simulator pose is used to command a successful placement. Stop cancels execution first; only after its worker exits and the arm is healthy may the session clear the task cancellation latch, preserving objects, held state, dialogue and unfinished obligations.

## Verification and implementation

- [x] Reproduce 20–79 mm stopped-release acceptance and stopped-but-permanently-locked state.
- [x] Add regression coverage for high, stale, rotated, moving and misleading-success release feedback; retain near-surface release.
- [x] Set interactive flat and stack clearances to 4 mm with depth error allowance; remove incompatible legacy per-label lowering in this task.
- [x] Check actual pose and stationarity immediately before opening, including recovery and held continuation.
- [x] Restore interaction after safe worker termination; reject concurrent old/new execution and preserve real faults.
- [x] Run real-contact camera-to-grasp-to-place tests recording gap and speed immediately before opening; run stop and UI regressions.
- [x] Back up the current live state, deploy to the existing service and verify unlocked interaction without claiming interrupted goals complete.

## Reproduction evidence

The old skill accepts actual TCP at 2, 20 and 79 mm above its release target. The recent banana-on-yellow release logged success with 0.194 mm TCP error but a nominal 6 mm bottom clearance. A live task stop returned to `busy=false`, `status=stopped`, `needs_reset=true`, `editor.editable=false`; the scene was not holding an object. The latter is an explicit reset latch, not an unfinished moving worker.
