แกหแกห

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
