# Pear transport and held placement continuation

## Incident and reproduced mechanism

The pear was still physically held by both fingers. It had not fallen. Its placement search exhausted the computation budget, then an explicit active observation failed with `held_feedback_changed` and latched a fault. The measured opening during otherwise successful held checks varied between approximately 38.476 and 38.616 mm. Active observation required opening equality within 0.001 mm, so ordinary contact settling invalidated the route.

The placement search also repeatedly restarted. In an offline clone of the saved contact scene, its first three candidates failed the final leaned-footprint check: the complete pear envelope exceeded the workspace Y limit by approximately 2.6–3.8 mm. Those are legitimate geometric refusals. The fourth candidate, at object center `[0.17, 0.325]`, passed geometry, reachability and fresh RGB-D support visibility. Reaching that result required approximately 10 seconds, but each computation slice stopped after 8 seconds and discarded completed rejections.

After the observation bug was fixed, replay exposed another failure: `place_held` synthesized a vertical lift using the current camera-view orientation, without planning that entry segment. The target at the raised height was unreachable, even though the independently checked destination pre-place/release/retreat poses were reachable.

The first live continuation also exposed a prepared-session state gap absent from the initial offline replay: active inspection left the session `ready`, and publishing each stationary camera scene reset it to `ready` again. The physics watcher therefore treated commanded motion of the carried pear as an external change. The guarded action stopped with `scene_changed` while retaining the pear. Active inspection now remains `executing` throughout its camera publications and motions, and restores `ready`/`stopped` in a `finally` block. Moved neighboring objects still invalidate the route; real grip loss still stops it.

## Changes

- Active observation accepts an opening variation of ±1 mm only within a route checked for that whole interval. Each finger collision box grows by 0.5 mm in either lateral direction. A valid grip outside that envelope requests another plan; missing feedback, actual loss, stop and arm faults still block motion.
- Release replanning can retain fully rejected candidates across computation slices of the same camera measurement and robot pose. An interrupted candidate is never marked rejected. One additional slice is allowed when the first slice completed useful rejection work. The state is local to each capture and is discarded before another view or changed scene; successful plans still undergo all final checks.
- MuJoCo held continuation now plans the actual current-TCP-to-pre-place route using fresh RGB-D neighbors and the existing swept-object/finger and linear-reach checks. Each executed segment checks scene revision, opening envelope and physical held feedback. It replaces the unplanned vertical-lift/P2P entry. The original measured grasp transform stays fixed.
- Missing camera frames cause a recoverable hold/reobserve result. Feedback is checked again after planning even when no route is returned, so a real drop cannot be mislabeled as a recoverable placement refusal.

The 4 mm release allowance, fresh actual-TCP gate and stationary opening checks from the preceding release fix remain in force. No friction, success tolerance or collision/visibility requirement was relaxed.

## Validation

- Regression tests reproduced the old failures before applying their fixes. Coverage includes normal opening fluctuations, expanded finger collision geometry, interrupted search continuation, capture/intervention invalidation, checked entry waypoints, cancellation, missing feedback and real grip loss.
- An independent selected group of observation, release, entry-route, inspection-feedback and stop tests passed 119 tests. A further release/held-planning/runner/stop/posture group passed 92 tests. These groups overlap; their counts are not additive.
- The physical dropped-object integration passed: opening the simulated physical fingers during held continuation causes failure and clears the held recovery record rather than claiming placement success.
- Prepared-session observation tests include camera publication resetting the session status, commanded carried-object movement, moved obstacles, dropped grip and cancellation. These specifically cover the difference between the offline replay and a live initialized session.
- After the prepared-session correction, a fresh selected group passed 60 tests. A real MuJoCo clone of the latest held scene with `_prepared=True` and an explicit expected revision completed two checked observation views in 26.03 seconds, retained real contact with the pear and kept its environment revision unchanged.
- The actual incident's physical checkpoint was restored offline with opposed finger joint signs and the existing closing-servo preload. Over 2 seconds of natural integration it remained held by real contacts: maximum pear displacement 0.310 mm, rotation change 0.0464 degrees and TCP displacement 0.229 mm. No attachment/weld was used.
- Full offline replay through `SkillExecutor`: active observation acquired a new view, the checked held entry route completed, the pear was released near the support into the green region, and both visual and physical verification passed. Final pear tilt was approximately 0.0053 degrees; contact, resting stability and upright posture checks passed. Physical execution took 16.173 simulated seconds; placement took 117.625 wall-clock seconds including planning and final observation.

## Replay and deployment limits

Private scene/chat/checkpoint files stay under the runtime checkout's `.git/pear-transport-20261119/` and are not committed. The saved API state and decision log contain the original measured dimensions and grasp pose but do not export the original private RGB-D point cloud. Replay therefore uses the recorded measured dimensions and a conservative geometric envelope; it does not synthesize point-cloud samples or use simulator object truth to plan. The original logged grasp angle and coordinates were rounded to four decimal places (at most 0.0029 degrees and 0.05 mm per coordinate).

This validates the saved pear transport/placement failure and safety regressions. It does not establish success for every possible fruit layout or remove the existing cost of final camera verification.

## Live deployment

The repaired service on port 8767 restored the latest held scene without resetting it. After the prepared-session correction, it executed only the existing pear observe/place/verify subplan through the normal skill executor. All three steps passed. The actual pear is released and upright in `zone_green`; fresh camera verification and physical contact, stability and posture checks passed (reported tilt 0.01 degrees).

The dashboard is idle with `needs_reset=false`, no held object and continuation available. All 39 chat messages, six original goals and the previous five history entries were preserved; three successful recovery entries were appended. The overall task remains stopped/incomplete so the other original goals can continue explicitly. Model-request count stayed unchanged: deployment and validation made no additional model/API requests. The page and live camera both returned HTTP 200. Private final state is saved in the runtime `.git/pear-transport-20261119/after-recovery.json`.
