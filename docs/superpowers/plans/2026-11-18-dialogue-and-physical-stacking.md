# Dialogue context and physical stacking implementation plan

**Goal:** retain bounded conversational meaning and execute camera-grounded object-on-object tasks with real MuJoCo contact dynamics and measured failure feedback.

**Decision:** the user selected MuJoCo on 2026-11-18. Reuse the existing full PiPER MuJoCo arm and force-limited contact gripper. The dashboard may render a geometry mirror; the mirror never attaches objects or decides physical outcomes. Keep the original kinematic backend available explicitly for old experiments.

**Architecture:** four independent units cover bounded dialogue, MuJoCo/dashboard adaptation, layered RGB-D perception, and semantic stacking execution. Goals reference a support label, never model-generated coordinates. Planning uses fresh camera measurements; physics state is reserved for simulation and independent evaluation. A robust flat support is required by the first stacking planner; curved supports must return a concrete limitation rather than fabricated success.

## Work

- [x] Reproduce lost clarification context; pass at most four completed semantic dialogue rounds to both API styles. Reset the semantic boundary without deleting visible chat. Preserve fresh-scene checks and explicit risk approvals.
- [x] Add `on_object` goals, disallow self-support and cycles, preserve immutable goals when supports are temporarily occluded, and show object tops separately from colored zones.
- [x] Adapt the existing MuJoCo world, arm and contact gripper into dashboard-compatible geometry, editing and camera interfaces. Install in an isolated local environment. Verify gravity, real contacts, stable support and a genuine dropped object.
- [x] Infer elevated support planes and object-relative height from RGB-D alone. Retain completeness and ambiguity checks; do not use simulator poses or labels as detections.
- [x] Plan release at the measured support top. Permit only intentional target/support contact, retaining full gripper and other-object collision checks. Reobserve before release and verify contact height, stability and specified support after settling.
- [x] Advance physics while idle. Unexpected physical motion invalidates cached observations/completion evidence. Report slip, fall, rolling and unstable release to bounded recovery.
- [x] Run offline protocol, conversation, geometry, perception and physical integration regressions, including apple on a red block, removing the support and recovery feedback. Record actual limitations and elapsed times.
- [x] Commit by feature, push, save the current scene and dialogue, deploy on port 8767 and verify live camera/model configuration without paid test calls.

## Acceptance boundaries

An accepted model JSON is never a successful placement. Both fresh camera evidence and independent physical support/stability must pass. No welded object attachment, manually assigned successful final pose, or deletion of the base collision geometry may stand in for a contact grasp. Any unsupported curved base, occluded identity, reach failure, slip or computation limit remains an explicit failure for recovery or human input.
