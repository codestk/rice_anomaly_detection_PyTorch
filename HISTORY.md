แกหแกห

## 2026-10-20

- `onefileV21.py`: YOLO Capture Classes now save as many images per object as a normal detection. The user saw fewer captures than in normal detect mode for the same object. Cause: capture had a per-class cooldown (`class_capture_min_interval`, "Capture Interval", default 0.04s), while normal detection auto-save saves every anomaly frame (`is_new_anomaly_found` is True on every anomaly frame, so `auto_save_min_interval` never applies). `_enqueue_class_capture` now passes `0.0` as the interval, so every frame showing the class is saved.
- "Capture Interval" still applies to the ejector-trigger capture and Reconstruction training samples. Tooltip and `set_capture_interval` docstring updated.
- Remaining known limit (unchanged): the shared save queue is `maxsize=16`, and `put_nowait` silently drops tasks when it is full. Both capture and detection saves are affected the same way.
- Files modified: `onefileV21.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV21.py` → `COMPILE_OK`. Not tested on the live rail.

## 2026-10-16

- `onefileV20.py`: the danger delayed snapshot is now a burst, so there are more danger images to train on. Before, one raw frame was saved 1s after a Dangerous-class hit. Now 10 raw frames are saved (`MainWindow.danger_delayed_burst_count = 10`), 150ms apart (`danger_delayed_burst_interval_ms = 150`), starting 1s after the hit. That covers about 1.0–2.35s, which ends before the Auto cleansing fan starts at 3s.
- Files are saved in `output/danger_delayed/` as `danger_delayed_<event timestamp>_<01..10>.png`. All frames in one burst use the same event timestamp. While paused, the live view keeps updating `current_frame`, so each step saves a newer camera frame. If detection stops partway through, the burst stops and the status bar shows how many frames were saved.
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`. Not tested on the live rail.

## 2026-10-14 (3)

- `onefileV20.py`: the servo (Pin 9) now moves **only for YOLO Dangerous Classes**, in every mode. `servo_detection_active` changed from `is_anomaly_present or dangerous_hit` to `dangerous_hit`, so Color/Recon/normal YOLO anomalies no longer open it; Auto-clear still closes it once no dangerous class is in frame for the clear delay.
- `reset_counter` (run on Start Detection) now sends the servo Clear only if the arm was actually left open (`_arduino_last_signal == "trigger"`), so starting detection no longer moves the servo by itself.
- Checkbox "Enable trigger on anomaly" renamed to "Enable servo on danger" with a tooltip. Settings key `arduino_enabled` unchanged.
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-14 (2)

- `onefileV20.py`: Auto cleansing ejector blast shortened from 1s to 0.5s (`danger_ejector_blast_ms = 500`). Tooltip updated, and the status-bar message now formats the blast length with `:g` so it shows "0.5s" instead of rounding to "0s".
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-14

- `onefileV20.py`: the "Reconstruction/Model" mode (`detector.mode == 'recon'`) is now collect-only. In `DetectionWorker._process_frame_impl`, a recon detection is saved to `output/recon_training/` (annotated) and `output/recon_training_original/` (raw) as `recon_<timestamp>.png`, counted in Detections and the summary, and then `is_anomaly_present` is cleared so nothing downstream acts on it — no servo, feeder stop, pause, beep, service breaker, or auto-save popup. Saves are rate-limited by the "Capture Interval (s)" setting. Random Save for Training still skips these frames (`recon_detected`).
- Only the Reconstruction mode is affected. In "Hybrid (OR)" the recon part still triggers actions as before, since that is what catches untrained objects for "Stop feeder on anomaly". YOLO does not run in Reconstruction mode, so Danger/Ejector/Capture classes are inactive there (unchanged).
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.
- Follow-up (same day): user still saw the servo arm move in Reconstruction mode. Detections no longer triggered it, but two non-detection paths still sent servo commands: the forced Clear in `reset_counter(force_clear=True)` on Start Detection, and the servo Auto-clear branch closing an arm left open from an earlier run. In Reconstruction mode both are now skipped (`recon_mode` guard in `_process_frame_impl` and `reset_counter`), and any pending servo trigger timer is canceled, so the servo receives no commands at all. The manual "Test Trigger / Send Clear" buttons still work.

## 2026-10-13 (2)

- `onefileV20.py` Auto cleansing: the ejector blast no longer starts together with the fan. New `MainWindow.danger_ejector_delay_ms = 3000`; `_start_danger_fan_clear` turns the fan on and schedules new `_fire_danger_ejector_blast(sequence_id)` 3s later, which checks the sequence id is still current (so a canceled sequence does not fire) and then fires the ejector for `danger_ejector_blast_ms`.
- Kept the user's own edits to the timings: `danger_fan_clear_ms = 10000` and `danger_ejector_blast_ms = 1000`. Tooltip updated to match (fan 10s, 1s blast 3s after the fan starts).
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-13

- `onefileV20.py`: the "Auto cleansing" danger sequence now also fires the air ejector for 2s (`MainWindow.danger_ejector_blast_ms = 2000`) at the moment the fan turns on, in `_start_danger_fan_clear` (`send_relay_trigger_test()` then `QTimer.singleShot(..., send_relay_clear_now)`). This is a deliberate change from 2026-09-23, when the user wanted the ejector idle during auto clear; the forced ejector-off at the start of the danger pause is unchanged, so the ejector only fires in this 2s window.
- The blast uses the configured Ejector ON/OFF commands and fires even if "Enable air ejector on anomaly" is unticked (same as the fan, which has no enable checkbox). The OFF is sent unconditionally after 2s, and `_cancel_danger_clear_sequence(stop_fan=True)` also sends ejector OFF.
- Updated the "Auto cleansing" tooltip. Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-11

- `onefileV20.py`: settings are no longer saved automatically when the window closes — removed `self._save_settings()` from `MainWindow.closeEvent`. Added a "Save Settings" button at the end of the bottom button row (after "Clear ALL output"), wired to new `_save_settings_clicked`, which calls `_save_settings()` and shows "Settings saved to config.ini." (or the error) in the status bar. Any change not saved with the button is lost on close.
- Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-10 (2)

- "Stop feeder on anomaly" in `onefileV20.py` now stops on **every** detection, so the user can collect samples for retraining: in `_process_frame_impl`, when `stop_feeder_on_detect` is on and the frame has any YOLO box (`yolo_labels` non-empty — normal, Capture-only, or Air Ejector-only), `is_anomaly`/`is_new_anomaly_found` are set so the existing FEEDOFF + 5 beeps + ejector force-clear + pause path runs. Color/Recon anomalies and Dangerous classes behave as before.
- Side effect, intended: with this checkbox on, Ejector-class objects pause the rail instead of being blown off (`general_pause_triggered` force-clears the ejector), so the sample stays on the rail to be picked up.
- Every such stop now saves a snapshot to `output/detections/` (with class names in the filename) even when Auto-save is off: the danger-only fallback save became `elif danger_event_started or general_pause_triggered:`.
- YOLO classes excluded by the YOLO Class Filter are still dropped inside `_run_yolo_detection` and never count.
- Updated the checkbox tooltip. Files modified: `onefileV20.py`, `HISTORY.md`. Verified with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-10

- **Reverted by the user (same day) — not in the code.** The checkbox split below was undone by the user restoring `onefileV20.py`; the file is back to a single "Stop feeder on anomaly" checkbox that does FEEDOFF + 5 beeps + ejector force-clear + pause. Do not re-add the split unless asked.
- Split the old "Stop feeder on anomaly" checkbox in `onefileV20.py` into two independent checkboxes in General Controls, per user request:
  - **Stop feeder on anomaly** (`stop_feeder_on_detect`) — now only sends FEEDOFF; detection keeps running.
  - **Pause detection on anomaly** (new `pause_on_detect`) — beeps 5 times, force-clears the ejector (`general_pause_triggered`), and pauses detection until the operator clicks Resume Detection.
  - Either can be ticked alone or both together (both = the old behavior). Dangerous classes are still excluded from both and handled by the Danger sequence.
- New `DetectionWorker.set_pause_on_detect` slot, wired in `_setup_detection_thread`; persisted to `config.ini` as `pause_on_detect`. When that key is missing, it defaults to the saved `stop_feed_on_detect` value, so existing setups that had the old checkbox ticked keep pausing as before.
- The detection popup (`emit_popup_snapshot`) now shows when either checkbox is ticked. Status message on pause no longer claims "feeder stopped", since the feeder may still be running.
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-07 (3)

- **Ejector Cooldown was intentionally reverted by the user — do not re-add.** The entries "2026-10-07" and "2026-10-07 (2)" below (`relay_cooldown_seconds`, `_relay_cooldown_until`, `ejector_cooldown_pause_requested`, the "Ejector Cooldown (s)" UI control) describe code that is no longer in `onefileV20.py`: the user tested it, it did not work in practice, and they restored the file to its 2026-10-04 state. The clumped-grain false positive after an air blast was solved instead by retraining the YOLO model with images captured during ejector firing as hard negatives, which works.
- Bug reported: when an untrained/unknown object (e.g. a stray screw, "หัวน๊อต") triggers "Stop feeder on anomaly" (general anomaly pause, not Danger), the air ejector was sometimes left stuck firing air continuously for the whole pause. Root cause confirmed: the ejector's own auto-clear only runs via per-frame polling (the `elif ... and not relay_detection_active:` branch), and once "Stop feeder on anomaly" pauses detection, no more frames get processed — so if an Ejector-class object happened to be in the same (or a very recent) frame as the unrelated general anomaly, the ejector could fire or stay mid-fire with nothing left to ever clear it until the operator manually clicks Resume Detection. This is the same class of bug fixed for Dangerous classes on 2026-09-23 (2), just never extended to the general "Stop feeder on anomaly" pause path.
- Fix: added a `general_pause_triggered` flag in `DetectionWorker._process_frame_impl`, set when `stop_feeder_on_detect` fires a pause this frame (mirrors `dangerous_hit`). The ejector's synchronous force-clear condition changed from `if dangerous_hit:` to `if dangerous_hit or general_pause_triggered:`, so a general-anomaly pause now force-clears/blocks the ejector in the very same frame, exactly like a dangerous-class hit already did.
- Added the same redundant GUI-side safety net already used for Danger (`send_relay_clear_now()`) to `MainWindow._handle_pause_detection_request` (Stop feeder on anomaly) and `_handle_service_breaker_trigger` (Service Breaker) — both pause detection and both could previously leave a mid-fire ejector stuck on for the same underlying reason.
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-07 (2)

- Correction to the Ejector Cooldown added earlier the same day: user clarified the intent was never just "block the ejector from re-firing" — it should actually **pause detection** for ~1-2s after the ejector fires, then auto-resume, not merely gate the relay trigger while everything else keeps running.
- Added `DetectionWorker.ejector_cooldown_pause_requested = pyqtSignal(float)` in `onefileV20.py`, emitted (carrying `relay_cooldown_seconds`) right alongside the existing `_relay_cooldown_until`/`_schedule_relay_trigger()` call when the ejector newly fires.
- `MainWindow._setup_detection_thread` connects it to new `_handle_ejector_cooldown_pause(duration_seconds)`: if detection is running, not already paused for another reason (manual-resume anomaly pause, danger sequence, etc.), and no danger auto-clear sequence is active, it calls `_pause_detection()` and schedules `_finish_ejector_cooldown_pause` via `QTimer.singleShot(duration_ms, ...)` to auto-resume — guarded by a `_ejector_cooldown_sequence_id` counter (same pattern as the existing danger-clear sequence id) so a stale timer can't resume a detection run that moved on. If detection was already paused for a different reason when the ejector fires, this handler intentionally no-ops rather than auto-resuming a pause it didn't start (e.g. never auto-resumes a "Stop feeder on anomaly" pause that needs a manual Resume click).
- The earlier `_relay_cooldown_until` same-frame gate is kept as a belt-and-braces check (mirrors the existing dangerous-hit race fix from 2026-09-23 (2)) — it blocks a worker-thread re-trigger in the same or next frame before the cross-thread pause signal has actually landed and stopped frame processing.
- Updated the "Ejector Cooldown (s)" tooltip to describe the real behavior (detection auto-pauses and resumes, not just "blocks re-fire").
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-07

- Problem reported: after the air ejector fires, the blast scatters grains into a clump that the YOLO model sometimes misreads as a new anomaly (e.g. "Rice_worm_rest" at ~0.72-0.79 confidence, screenshot showed two such false boxes), re-firing the ejector on its own mess and scattering it further in a loop.
- Added an "Ejector Cooldown" to `onefileV20.py`: `DetectionWorker.relay_cooldown_seconds` (default 1.5s) + `self._relay_cooldown_until` timestamp. After the ejector fires, `_relay_cooldown_until` is set to `now + relay_cooldown_seconds`; the trigger-scheduling condition in `_process_frame_impl`'s ejector block now also requires `now_ts >= self._relay_cooldown_until`, so no new trigger can fire during that window even if `relay_detection_active` flickers back True (e.g. from borderline confidence on the disturbed clump). The existing per-event dedupe (`_relay_last_signal != "trigger"`) still governs everything else; this cooldown only adds a floor under how soon a *new* trigger can start after the last one. Auto-clear behavior (turning the ejector back off) is untouched.
- New UI control "Ejector Cooldown (s)" (a `DelayControl`, range 0.0-10.0s, step 0.1, default 1.5) added on the same row as "Capture Interval (s)" in Model & Detection Thresholds, wired to `_update_relay_cooldown` → `DetectionWorker.set_relay_cooldown`, pushed on thread setup, and persisted to `config.ini` (`relay_cooldown`). `_relay_cooldown_until` is reset to 0 in `reset_counter`.
- This only prevents the *re-trigger loop*; it does not fix the underlying false positive itself. Recommended follow-up (not yet done): collect the false-positive images already being saved to `output/ejector_captures/` and add them as hard-negative training examples so the model learns to tell a disturbed grain clump apart from a real worm.
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-10-04

- Moved the "White Balance (Once)" button + hint label in `onefileV20.py` from the left column's camera/exposure block (below the Exposure hint) into the "General Controls" group box, as a new row right after the "Mode:" dropdown — per a screenshot where the user drew an arrow from the button to its new target spot.
- Follow-up correction: user clarified it should share the SAME row as "Mode:", not its own row below it — merged into `mode_row` (Mode label + combo, then White Balance button + hint, all in one `QHBoxLayout`) instead of a separate row.
- Removed the `self.exposure_hint_label` QLabel ("-13 = Fast/Dark, 0 = Slow/Bright") entirely, per a screenshot with a red arrow pointing at it — it duplicated info already in the Exposure slider's own value label (which shows e.g. "-5 (Balanced)" via `_format_exposure_label`), so it was redundant.
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-09-29

- **Work now targets `onefileV20.py` instead of `onefileV19.py`** per user request (V20 was already a full copy including every fix through 2026-09-28 (2)).
- Added a separate, independent confidence threshold just for the Air Ejector: new `AnomalyDetector.yolo_relay_confidence` (default 0.75, deliberately higher than the general `yolo_confidence` default 0.35) with setter `set_yolo_relay_confidence`. Reason: this camera's higher resolution makes for noisier/harder-to-train detections, so the user wants the ejector to require a stricter confidence than the general "YOLO Confidence" slider before it's allowed to fire, even on a box that's confident enough to count as a normal anomaly elsewhere.
- New UI slider "YOLO Air Ejector Confidence" added right under "YOLO Air Ejector Classes" in the Model & Detection Thresholds group, with its own label/tooltip, wired to `_update_yolo_relay_confidence`, and persisted to `config.ini` (`yolo_relay_confidence`).
- Plumbing: `AnomalyDetector` now tracks `last_yolo_confidences` (parallel array to `last_yolo_detections`/`last_yolo_class_ids`, populated in `_apply_yolo_results` and cleared in the same places as those two). `DetectionWorker._get_relay_yolo_hits` signature changed from `(labels, class_ids)` to `(labels, class_ids, confidences)` and now rejects any relay-class match whose confidence is below `yolo_relay_confidence`, independent of whatever confidence let it through the general YOLO Class Filter. This was necessary because the existing relay-hit check only matched on label/class-id text with no confidence awareness at all, so a per-instance confidence gate had to be threaded through explicitly rather than reusing the general per-box gate already applied inside `_run_yolo_detection`.
- Files modified: `onefileV20.py`, `HISTORY.md`.
- UPDATE: `python -m py_compile onefileV20.py` was blocked by a transient sandbox tool error for the rest of that session; it was confirmed passing (`COMPILE_OK`) at the start of the next session (2026-09-30), after the capture-interval change below was also applied.

## 2026-09-30

- Exposed the shared "capture interval" cooldown (`DetectionWorker.class_capture_min_interval`, governs both "YOLO Capture Classes" saves and the ejector-trigger capture) as a UI control instead of a hardcoded value — user wanted to tune it themselves without asking for a code change each time.
- Added `DetectionWorker.set_capture_interval(value)` (plain method, called directly like `configure_arduino`/`configure_relay`, not a signal-marshaled pyqtSlot).
- New UI row "Capture Interval (s):" (a `DelayControl` spin box, range 0.0-2.0s, step 0.01, 2 decimals, default 0.04) added under the new "YOLO Air Ejector Confidence" slider in the Model & Detection Thresholds group, wired to `_update_capture_interval`, pushed to the worker on thread setup and after settings load, and persisted to `config.ini` (`capture_interval`).
- Files modified: `onefileV20.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV20.py` → `COMPILE_OK`.

## 2026-09-28 (2)

- Colored the "Feeder ON (Pin 7)" and "Feeder OFF (Pin 7)" buttons in the Arduino Integration panel of `onefileV19.py` — green for ON, red for OFF — so they're easier to tell apart at a glance (user pointed at these two buttons in a screenshot). Other Arduino test buttons (Servo Trigger/Clear, Fan ON/OFF) are unchanged.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-28

- Lowered `DetectionWorker.class_capture_min_interval` in `onefileV19.py` further, from 0.1s to 0.04s. User reported the new HuaTeng camera only produced 4 "YOLO Capture Classes" images per object crossing vs 8 with the old camera, despite the new camera's higher rated FPS. Root cause: capture count is governed by (object dwell time in frame) ÷ (cooldown), not by the camera's max FPS — the new camera's field of view/framing shows each object for less real time (~0.3-0.4s vs ~0.7-0.8s on the old camera), so the same 0.1s cooldown yielded fewer shots even though the achieved processing FPS (~32fps) was higher. 0.04s keeps pace with that ~32fps throughput and should recover a similar shot count within the shorter dwell window.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-27

- Lowered `DetectionWorker.class_capture_min_interval` in `onefileV19.py` from 0.8s to 0.1s, applied uniformly to every class (same shared cooldown value for all labels, as before). Reason: an object crossing the rail is visible for ~0.3-0.8s, so the old 0.8s per-label cooldown let only the first frame through and skipped the rest — "YOLO Capture Classes" (`output/yolo_captures/`) ended up with exactly one image per object instead of the 5-8 a full crossing produces. At 0.1s the same object now yields several captures across its crossing.
- This interval is shared with the ejector-trigger capture (`_enqueue_relay_capture`, added 2026-09-23), but that path is already gated by the trigger-state transition (only fires once per idle→trigger event) so it is not meaningfully affected by the lower value.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-26

- `onefileV19.py`: when the air ejector newly fires on an Ejector-class object, it now also counts as a detection — `anomaly_count` +1, the class is recorded in the detection summary, and the beep plays if "Enable Beep Sound" is on. Skipped if the same frame was already counted as a normal anomaly (no double count). It does NOT trigger Servo, "Stop feeder on anomaly", or auto-save; the ejector stays independent of those.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-23 (3)

- Added an "which image caused this ejector shot" trail: whenever the air ejector newly fires (relay transitions from idle to trigger), `DetectionWorker._process_frame_impl` in `onefileV19.py` now saves that frame to `output/ejector_captures/` (annotated) and `output/ejector_captures_original/` (raw), one file per matched class, named `<ClassName>_<timestamp>.png` — e.g. `Stone_20260923_...png`. Mirrors the existing "YOLO Capture Classes" save mechanism.
- Refactored `DetectionWorker._enqueue_class_capture` into a shared `_enqueue_labeled_capture(..., capture_dir, original_dir, rate_limit_state, min_interval, source)` helper, reused by the existing capture-classes path and the new `_enqueue_relay_capture`. Added `self._relay_capture_last_save_time = {}`.
- Save-worker log line for this task kind now reads `"{source} '{label}' saved: ..."` (`source` is `"YOLO capture class"` or `"Ejector trigger"`) instead of a hardcoded "YOLO capture class" label, so the console log also distinguishes the two.
- `_process_frame_impl` now keeps the `relay_hits` list (from `_get_relay_yolo_hits`) around instead of discarding it, so the triggering class name(s) are available to pass into the capture.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-23 (2)

- Closed a race in the 2026-09-23 ejector/auto-cleansing fix: on the exact frame where a dangerous class is first detected, `DetectionWorker._process_frame_impl` in `onefileV19.py` used to run its ejector trigger logic *before* the (cross-thread, queued) `dangerous_detection_triggered` signal reached `MainWindow._handle_dangerous_detection`, so the ejector could still fire for one frame if that same frame also matched an Ejector class, before being cleared a moment later.
- Now `dangerous_hit` takes priority over the ejector's own trigger check, synchronously on the worker thread: whenever a dangerous class is present in the frame, any active/pending ejector trigger is force-cleared immediately and no new trigger is scheduled, regardless of Ejector Classes. The `send_relay_clear_now()` call added on 2026-09-23 in `_handle_dangerous_detection` stays as a second, redundant safety net.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-23

- Fixed: air ejector could stay/latch active during a dangerous-YOLO-class pause + "Auto cleansing" fan-clear sequence, since no frames are processed while detection is paused so nothing sent a clear command. `MainWindow._handle_dangerous_detection` in `onefileV19.py` now calls `DetectionWorker.send_relay_clear_now()` right after pausing (cancels any pending relay trigger timer and force-sends the ejector OFF command), so the ejector stays idle for the whole danger pause/auto-cleansing window.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-21

- Renamed the "Stop feeder on detection" checkbox label to "Stop feeder on anomaly" in `onefileV19.py` (tooltip and underlying `stop_feed_on_detect_check`/`stop_feeder_on_detect` names unchanged).
- Removed the always-visible Thai hint label added under that checkbox on 2026-09-20 (2) — user said it took up too much space; the hover tooltip still explains the pause/5-beep/manual-Resume behavior.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-20 (2)

- Added a visible (always-shown) Thai-language hint label under the "Stop feeder on detection" checkbox in `onefileV19.py`, explaining the pause/5-beep/manual-Resume behavior directly in the UI instead of relying only on the checkbox's hover tooltip.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-09-20

- Changed `onefileV19.py` "Stop feeder on detection" behavior: on any new anomaly (including objects the model has never seen before), it now sends FEEDOFF, plays a bounded 5-beep alert, and **pauses** detection instead of fully stopping it, so an operator can remove the item and click Resume Detection to continue without reopening the camera.
- Replaced `DetectionWorker.stop_detection_requested`/`_handle_stop_detection_request` (full stop) with `pause_detection_requested`/`_handle_pause_detection_request` (pause) in `MainWindow`.
- Removed the now-dead continuous stop-feed beep machinery (`_start_stop_feed_beep`, `_continuous_beep_loop`, `_stop_stop_feed_beep`, `_stop_feed_beep_active`, `_stop_feed_beep_thread`) and its call sites, replacing it with `_play_pause_on_anomaly_beep` (5 beeps, ~0.55s apart, self-terminating).
- Updated the "Stop feeder on detection" checkbox tooltip to describe the new pause/beep-5x/manual-resume behavior.
- Files modified: `onefileV19.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV19.py`.

## 2026-08-24

- Updated `onefileV18.py` dangerous YOLO class handling so after a danger class triggers feeder OFF and detection pause, the app waits 1 second then saves the latest live camera frame.
- Added a separate `output/danger_delayed` folder for these delayed danger-class snapshots.
- Removed the delayed snapshot hook from service breaker handling; service breaker now keeps its previous popup/snapshot behavior only.
- Updated Clear ALL output recreation to include the new delayed danger snapshot folder.
- Files modified: `onefileV18.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV18.py`.

## 2026-08-23

- Updated `onefileV18.py` monitor camera handling so the monitor camera always uses normal OS/OpenCV camera enumeration instead of the HuaTeng SDK camera list.
- Monitor live view now starts with the normal `DSHOW` backend and fixed `NV12` settings, independent of the detection camera backend selection.
- Changed the fixed monitor FourCC from `MJPG` to `NV12`.
- Adjusted same-index camera conflict checks so a HuaTeng SDK detection camera does not block a normal monitor camera with the same combo index.
- Rolled back a short-lived `Auto White Balance` checkbox experiment for the detection camera; the existing `White Balance (Once)` button remains.
- Added `20` FPS to the Frame Rate dropdown in `onefileV18.py`.
- Files modified: `onefileV18.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV18.py mvsdk_capture.py`.

## 2026-08-22

- Briefly changed `onefileV18.py` YOLO capture-only saving to combine multiple matched classes into one image set per frame.
- Rolled that combined-per-frame behavior back because it made output grouping unclear.
- Current behavior is restored to one image set per matched `YOLO Capture Classes` class, with a same-filename original copy under `output/yolo_captures_original`.
- Capture-only detections still do not contribute anomaly contours/triggers when they only pass because of `YOLO Capture Classes`.
- Files modified: `onefileV18.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV18.py`.

## 2026-08-21

- Investigated why `YOLO Capture Classes` in `onefileV18.py` did not save images to `output`.
- Found that capture-only classes were being removed by the active `YOLO Class Filter` before the capture save logic could see them.
- Updated YOLO filtering so classes listed in `YOLO Capture Classes` can pass through for capture saving even when they are not in `YOLO Class Filter`.
- Kept capture-only YOLO detections from contributing anomaly contours/triggers when they only passed because of `YOLO Capture Classes`.
- Added paired original-frame saving for YOLO capture classes under `output/yolo_captures_original` using the same filename as `output/yolo_captures`.
- Files modified: `onefileV18.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV18.py`.

## 2026-07-24

- Created and updated `onefileV16.py` based on `onefileV15.py`.
- Cleaned up dead code including commented-out `open_camera_by_index` blocks and placeholder comments.
- Added visual status LED indicators and labels for Feeder Relay (ON/OFF) and Fan Relay Pin 8 (ON/OFF) in the Arduino Integration panel.
- Added explicit Arduino Pin labels across UI hardware indicators and control buttons: `Servo (Pin 9)`, `Feeder (Pin 7)`, and `Fan (Pin 8)`.
- Connected `DetectionWorker` signals `feeder_state_changed` and `fan_state_changed` to update hardware status indicators in real time when Feeder or Fan commands are sent.
- Implemented `stop_save_worker()` in `DetectionWorker` and integrated it into `MainWindow.closeEvent()` to gracefully flush the save queue and join `_save_thread` on exit.
- Files modified: `onefileV16.py`, `HISTORY.md`.
- Verified syntax with `python -m py_compile onefileV16.py`.

## 2026-07-23

- Updated `onefileV15.py` paused live-view overlay placement so `LIVE VIEW ONLY - DETECTION PAUSED` is right-aligned near the top of the frame instead of the top-left corner.
- Moved focus overlay drawing before the paused label so it does not cover the paused status text.
- Files modified: `onefileV15.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-07-23

- Updated `onefileV15.py` paused live-view overlay color from yellow to green for better readability.
- Files modified: `onefileV15.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-07-22

- Updated `onefileV15.py` so pausing detection no longer pauses the detection camera capture thread.
- Added a `MainWindow.detection_frame_ready` routing signal so live camera frames are sent to `DetectionWorker` only when detection is not paused.
- During pause/auto-cleansing, the main Live Feed now shows raw real-time camera video with a `LIVE VIEW ONLY - DETECTION PAUSED` overlay, allowing the operator to watch the fan clear the foreign object without running model/YOLO/HSV detection.
- Throttled paused live display to about 30 FPS to avoid unnecessary UI load.
- Files modified: `onefileV15.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-07-01

- Updated `readme.txt` to be bilingual with the English guide first, followed by the Thai guide.
- Added English documentation for `onefileV15.py` covering app purpose, detection modes, monitor live view, Arduino integration, auto cleansing, output/config files, usage steps, and safety notes.
- Preserved the Thai user guide content after the English section.
- Files modified: `readme.txt`, `HISTORY.md`.

## 2026-07-01

- Updated `onefileV15.py` Last Detection Snapshot path label for readability.
- Increased the saved image path font from 9pt to 13pt, added stronger text weight, padding, and a minimum label height.
- Files modified: `onefileV15.py`, `HISTORY.md`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-06-30

- Rewrote `readme.txt` as a user-facing Thai guide for `onefileV15.py`.
- Documented app purpose, detection modes, YOLO, HSV, Arduino integration, monitor live view, auto cleansing, output folders, config values, hardware/software requirements, usage steps, and safety notes.
- Kept the resolution list and `check_blurry_png.py` command as reference material.
- Verified `onefileV15.py` still compiles with `python3 -m py_compile onefileV15.py`.

## 2026-06-30

- Updated `onefileV15.py` to add a separate `Monitor Exposure` slider for the live-only monitor camera.
- Monitor exposure now adjusts independently from the detection camera exposure slider and applies immediately while the monitor is running.
- Monitor capture remains fixed at `1280x720`, `30fps`, and `MJPG`; only monitor exposure is adjustable.
- Saved and restored `monitor_camera_exposure` in `config.ini`, defaulting to `0`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-06-30

- Updated `onefileV15.py` monitor camera settings to fixed capture values: `1280x720`, `30fps`, exposure `0`, and `MJPG`.
- Monitor camera no longer reads the shared Resolution, Frame Rate, or Exposure UI controls; those controls now affect only the detection camera.
- Updated the monitor status label to show the fixed monitor capture settings.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-06-30

- Updated `onefileV15.py` monitor camera startup to force `MJPG` FourCC for the live-only monitor stream.
- Detection camera FourCC selection remains controlled by the existing UI dropdown.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-06-30

- Updated `onefileV15.py` to add a live-only monitor camera for checking machine-room operation.
- Added a separate `Monitor Camera` selector, `Start Monitor` and `Stop Monitor` buttons, and a separate `Monitor Live View` window.
- Reused `VideoThread` for monitor capture but connected frames directly to the monitor display only; monitor frames do not enter model, YOLO, HSV, detection counters, or Arduino logic.
- Added guards so the running monitor camera cannot be reused as the detection camera at the same time, and the monitor thread stops when the monitor window or app closes.
- Saved and restored `monitor_camera_index` in `config.ini`.
- Verified syntax with `python3 -m py_compile onefileV15.py`.

## 2026-06-29

- Updated `onefileV14.py` danger fan auto-clear re-trigger handling.
- Fixed fan OFF then fan ON repeating a second cycle after auto-resume when the same dangerous class was still visible.
- Auto-resume after `FANOFF` now keeps the danger alarm armed until a no-danger frame is observed, while manual Resume still re-arms danger immediately.
- Added a guard so danger signals during an active auto-clear sequence wait for the current sequence instead of starting a new fan cycle.

## 2026-06-29

- Updated `onefileV14.py` danger fan auto-clear timing.
- Added a 3-second delay after `FEEDOFF` before sending `FANON`, then kept the existing 20-second fan clear, `FANOFF`, resume detection, 5-second delay, and `FEEDON` flow.
- Updated the checkbox tooltip to describe the new feeder-off-to-fan-on delay.

## 2026-06-29

- Updated `onefileV14.py` General Controls UI text.
- Renamed the danger fan auto-clear checkbox label from `auto cleansing` to `Auto`.

## 2026-06-29

- Updated `onefileV14.py` General Controls UI text.
- Renamed the danger fan auto-clear checkbox label to `auto cleansing` while keeping the existing `danger_auto_fan_clear` setting and behavior unchanged.

## 2026-06-29

- Updated `onefileV14.py` to add optional danger-class fan auto-clear.
- Added `Danger fan auto-clear` checkbox, saved as `danger_auto_fan_clear` in `config.ini`, defaulting off.
- When enabled and Arduino is connected, a dangerous YOLO class now pauses detection, keeps feeder off, runs `FANON` for 20 seconds, sends `FANOFF`, resumes detection, waits 5 seconds, then sends `FEEDON`.
- Added sequence guards so Stop/Start, option disable, Arduino disconnects, or stale timers do not accidentally resume detection or turn the feeder on.

## 2026-06-29

- Updated `onefileV14.py` danger-class pause/resume behavior.
- Added `DetectionWorker.reset_danger_alarm()` to re-arm dangerous YOLO class handling.
- `_resume_detection()` now resets both service breaker and danger alarm state so pressing Resume can pause/break detection again if a dangerous class is detected.
- Issue fixed: after the first danger-class pause, `_danger_alarm_active` stayed true across Resume and blocked later `dangerous_detection_triggered` signals until Stop/Start reset the worker counters.

## 2026-06-23

- Updated `onefileV13.py` Arduino controls to add fan ON/OFF buttons for the fan relay on Arduino pin 8.
- Added `DetectionWorker.send_arduino_fan_on()` and `send_arduino_fan_off()` methods that send `FANON` and `FANOFF` over the existing Arduino serial connection.
- Added `Fan ON (Pin 8)` and `Fan OFF` buttons next to the feeder controls, with enabled/disabled state tied to Arduino connection status.
- Updated `aduno/รวม  พัดลม v13` so plain `FANON` keeps the fan on until `FANOFF`, while `FAN` and `FANON <milliseconds>` still support timed fan runs.
- Verified syntax with `python3 -m py_compile onefileV13.py`.

## 2026-06-23

- Updated Arduino sketch `aduno/รวม  พัดลม` to add fan control on Arduino pin 8.
- Added serial commands:
  - `FAN` or `FANON [milliseconds]` turns the fan on and auto-stops after the configured delay.
  - `FANOFF` or `STOPFAN` turns the fan off immediately.
  - `SETFANDELAY <milliseconds>` changes the default fan delay, initially 5000 ms.
- Implemented the fan auto-stop with `millis()` timing instead of blocking `delay()` so servo and feeder serial commands remain responsive.
- Note: fan pin uses LOW = ON and HIGH = OFF, matching the feeder relay behavior in the existing sketch.

## 2026-06-12

- Investigated servo misfires when 2-3 anomalies are detected close together in `onefileV12.py`.
- Updated Arduino trigger handling in `DetectionWorker`:
  - Added a bounded pending-trigger queue so multiple new tripwire objects in the same/nearby frames are not collapsed into one servo event.
  - Counted multiple new tripwire centroids as separate anomaly events instead of one boolean event.
  - Allowed auto-clear to re-arm the servo when queued trigger events are waiting, while keeping the old clear-after-no-detection behavior when there is no queue.
  - Cleared pending trigger state when Arduino is disabled or detection counters reset.
- Verified syntax with `python3 -m py_compile onefileV12.py`.
- Note: For multiple close objects, Arduino Auto-clear should remain enabled and `arduino_clear_delay` should be tuned to the servo's real movement time.

## 2026-06-12 follow-up

- Replaced the queued multi-trigger experiment because it made the servo alternate trigger/clear too aggressively.
- Kept servo behavior as one trigger per active anomaly burst, then added clear hysteresis:
  - `_last_anomaly_seen_time` now tracks any servo-active detection.
  - Auto-clear waits until no servo-active detection has been seen for the configured clear delay.
  - This should reduce trigger/clear flapping when detection flickers between closely spaced objects.
- Verified syntax with `python3 -m py_compile onefileV12.py`.
