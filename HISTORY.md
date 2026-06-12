แกหแกห

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
