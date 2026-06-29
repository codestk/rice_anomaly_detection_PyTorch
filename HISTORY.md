แกหแกห

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
