# Low release and stop without scene reset

## Mechanism and change

The old release recovery accepted a TCP up to 80 mm above its target. An interrupted descent at 20 or 79 mm could therefore open and report completion. Even normal interactive placement commanded a 10 mm table clearance and 6 mm stack clearance. Opening trusted the motion result without independently checking the current pose.

The interactive task now commands 4 mm bottom-to-support clearance for flat table and stack destinations. Its legacy fruit-specific lowering is disabled so a 5 mm pear offset cannot drive below the surface. A first 2 mm experiment was rejected: the pear contacted the table before opening (actual gap -0.613 mm), slid 16.98 mm and failed the unchanged 15 mm point tolerance. With 4 mm allowance, the original point and green-zone integrations pass.

Normal, held continuation and stopped-near-target recovery share an opening gate. It requires fresh finite TCP feedback within 5 mm horizontally, 2 mm vertically and 3 degrees of the planned pose. Two checks 120 ms apart require low translational and angular drift. Abort and scene-change checks still apply immediately before opening. A mismatch retains the grip and reports an unconfirmed release; the former high-drop fallback is removed. This is camera/planned-support plus measured robot feedback, not simulator-object pose used for control.

Task stop previously latched both cancellation and `needs_reset`, leaving editing permanently disabled after the worker exited. Stop now holds the physical arm and cancels work first. A completion barrier and idempotent repeated stops prevent stop/rearm races. After the sole owning worker unwinds, the same healthy, enabled session with fresh finite feedback can clear the task abort latch without homing or opening the gripper. Real faults remain latched. An old blocking network request may finish its socket I/O, but cannot parse a decision, publish output, retry a POST or move the robot after cancellation.

Stopped tasks retain original goals and restoration records and can explicitly continue. New observations invalidate old scene IDs. Resuming after a stop must not replace the recorded original object positions with temporary positions; this separate regression is fixed. Sending an entirely new instruction still starts a new task; use continuation for an existing restoration commitment.

## Verification

- Red before fix: 20/79 mm interrupted descents opened; successful-motion reports with high/lateral/rotated/stale/invalid/moving feedback still opened. The new release group initially had 16 failures.
- Release feedback, skill and stack geometry: 66 passed.
- Local/strict/preferred release, pear posture and release observation: 59 passed.
- Stop, model, server, scene and orchestration groups: 223 passed in the delegated full group; an independent subset rerun passed 146 in 28.38 s. The final review also independently re-ran both stop races and the original-pose preservation regression (3 passed).
- UI rendering/listeners: 426 assertions across five suites.
- Original real-contact camera-to-pear pick/point and green-zone placement: 3 passed in 291.80 s. The strict point tolerance and upright check are retained.
- Additional real-contact low-release/rejection checks: 2 passed in 139.08 s. Banana release over a 50 mm block measured 3.762 mm bottom gap and 0.01465 mm/s object linear speed; actual contact/support remained stable after settling.

## Limits

The banana rolled onto its side on the narrow block and its final camera confirmation remained incomplete in that test. The API correctly retained `ok=false / placement_not_observed`; physical support alone is not a completed task. A pear placement whose leaned reachable pose cannot provide full finger/footprint clearance remains refused, without grasping or opening. These two checks do not claim autonomous completion of every stack. UI DOM/listener and HTTP behavior are verified; native browser inspection was unavailable because the existing window was minimized.

The live stopped scene is backed up privately under the runtime checkout's `.git/release-stop-20261119/`, including chat and unfinished task state. That private data is not committed. Deployment restores the saved physical scene and arm pose without automatically resuming or marking a goal complete.

Deployment checked the existing port 8767: `busy=false`, `needs_reset=false`, editing enabled and continuation available. All six objects, the 21 original chat messages and the interrupted pear-on-green goal were retained (one deployment notice appended). The largest physical-position change from the saved scene was below 0.001 mm. The page and live camera returned HTTP 200, and the served UI source matched the deployed file. Refresh the existing browser page to load its new button logic.
