Rice Anomaly Detection - onefileV15.py
======================================

Main application file: onefileV15.py

This application is a PyQt6 GUI for live anomaly detection from camera input. It uses PyTorch, OpenCV, optional YOLO detection, HSV color detection, and optional Arduino control for real machine actions such as servo trigger, feeder control, and fan clearing.


ENGLISH GUIDE
=============

1. What This Application Can Do
-------------------------------

1. Live anomaly detection
   - Opens a main detection camera.
   - Displays live processed video.
   - Shows FPS, image size, detection count, and focus measurement.

2. Multiple detection modes
   - Reconstruction/Model: uses an autoencoder model to detect frames that differ from normal samples.
   - Color (HSV): detects target color ranges using HSV thresholds.
   - Hybrid (OR): detects if either the reconstruction model or HSV color logic finds an anomaly.
   - YOLO Only: detects objects/classes directly with a YOLO model.

3. Multi-range HSV color detection
   - Supports Hue1, Hue2, and Hue3 target ranges.
   - Each Hue range can be enabled or disabled.
   - Users can click the image to sample HSV values from real camera frames.
   - Includes hue target lock and center marker tools for setup.

4. YOLO support
   - Load YOLO model files such as .pt or .onnx.
   - Enable/disable YOLO detection.
   - Adjust YOLO confidence.
   - Filter specific YOLO classes.
   - Configure dangerous classes that can pause detection and trigger feeder/fan actions.

5. Separate monitor live-view camera
   - A second camera can be used only for observing the machine room or machine interior.
   - The monitor camera does not enter the AI model, YOLO, HSV logic, counters, or Arduino trigger logic.
   - Fixed monitor capture settings: 1280x720, 30fps, MJPG.
   - Monitor exposure has its own separate slider.
   - The app prevents using the same camera for detection and monitor live view at the same time.

6. Arduino integration
   - Select serial port and baudrate.
   - Send trigger and clear commands for a servo or external controller.
   - Set trigger delay and clear delay.
   - Manual Feeder ON/OFF controls.
   - Manual Fan ON/OFF controls.
   - Displays Arduino connection and servo/trigger state.

7. Automatic stop and clearing behavior
   - Stop feeder on detection: sends FEEDOFF and stops detection when an anomaly is detected.
   - Service breaker: pauses detection after repeated back-to-back detections.
   - Auto cleansing: when a dangerous YOLO class is detected, the app pauses detection, stops the feeder, runs the fan, resumes detection, then turns the feeder back on.

8. Saving images and results
   - Auto-save detections.
   - Capture images during live detection.
   - Saves original and processed images to output folders.
   - Shows popup snapshots and summary dialogs for detection events.

9. Training and dataset preparation helpers
   - Includes a Train Model button for the local training pipeline.
   - Includes check_blurry_png.py for checking blurry images before training or labeling.


2. Benefits
-----------

- Helps detect abnormal rice grains, defects, or unwanted objects in real time.
- Reduces manual visual inspection.
- Improves consistency in sorting or inspection workflows.
- Can control real hardware through Arduino, such as feeders, servos, and fans.
- Provides a monitor-only camera for machine observation without adding AI workload.
- Allows threshold, HSV, YOLO, camera, and Arduino settings to be adjusted from the GUI.
- Saves detection images for review, reporting, or future dataset improvement.


3. Recommended Hardware and Software
------------------------------------

Recommended hardware:
- Windows PC.
- NVIDIA GPU if using PyTorch/YOLO on GPU.
- One USB camera for detection.
- Optional second USB camera for monitor live view.
- Optional Arduino for servo, feeder, or fan control.

Main Python packages:
- PyQt6
- opencv-python
- numpy
- torch
- torchvision
- ultralytics, if YOLO is used
- pyserial, if Arduino is used


4. How To Start
---------------

Open a terminal in the project folder and run:

    python onefileV15.py

If using a virtual environment, activate it first.


5. Basic Usage
--------------

1. Select a detection mode.
   - Reconstruction/Model requires loading a model first.
   - Color (HSV) can run without a model.
   - Hybrid (OR) combines model and HSV detection.
   - YOLO Only requires loading a YOLO model first.

2. Select the detection camera.
   - Click List Cameras.
   - Choose Source.
   - Choose Backend: Auto, MSMF, or DSHOW.
   - Choose FourCC, Frame Rate, and Resolution.
   - Adjust the detection camera Exposure slider.

3. Set detection thresholds.
   - MSE Threshold for reconstruction/model mode.
   - CV Threshold and Contour Area for contour filtering.
   - YOLO Confidence for YOLO detection.

4. Click Start Detection.
   - The app opens the detection camera.
   - Frames are sent to DetectionWorker.
   - Results appear in the Live Feed window.

5. During operation.
   - Pause Detection pauses the live detection stream.
   - Resume Detection continues detection.
   - Stop Detection stops the detection camera.
   - Capture Image saves the current live frame.


6. Monitor Live View
--------------------

The monitor camera is only for watching the machine room or machine interior. It is not used for AI detection.

How to use:
1. Click List Cameras.
2. Select Monitor Camera.
3. Adjust Monitor Exposure if the image is too bright or too dark.
4. Click Start Monitor.
5. A separate Monitor Live View window opens.
6. Click Stop Monitor or close the monitor window to stop the monitor camera.

Fixed monitor camera settings:
- Resolution: 1280x720
- Frame rate: 30fps
- FourCC: MJPG
- Exposure: adjustable through the Monitor Exposure slider

Notes:
- Do not use the same camera for detection and monitor live view while detection is running.
- For two USB cameras, use separate USB ports/controllers where possible.


7. HSV Detection
----------------

1. Select Color (HSV) or Hybrid (OR).
2. Enable Hue1, Hue2, or Hue3 as needed.
3. Adjust Hue Low/High, Saturation Min, and Value Min.
4. Click on the image to sample a color from the real camera frame.
5. Use Test Image to verify settings on still images before running live detection.

HSV detection is useful when defects or target objects have clear color differences.


8. YOLO Detection
-----------------

1. Click Browse YOLO...
2. Select a .pt or .onnx model file.
3. Enable YOLO detection.
4. Adjust YOLO Confidence.
5. Use YOLO Class Filter to detect only selected classes.
6. Use YOLO Dangerous Classes to define classes that should trigger pause/clearing behavior.

Example dangerous classes:

    Black_Head, 2

You can enter class names or class IDs depending on the YOLO model.


9. Arduino Usage
----------------

1. Connect Arduino to the PC.
2. Select Port and Baudrate.
3. Click Connect.
4. Configure:
   - Enable trigger on anomaly
   - Trigger Delay
   - Auto-clear
   - Clear Delay
   - Trigger Command
   - Clear Command
5. Use Test Trigger and Send Clear before production use.
6. Use Feeder ON/OFF and Fan ON/OFF for manual hardware testing.

Common commands:
- FEEDON
- FEEDOFF
- FANON
- FANOFF
- Trigger command, for example 1
- Clear command, for example 0


10. Auto Cleansing / Dangerous Class Flow
-----------------------------------------

When Auto cleansing is enabled and Arduino is connected:

1. YOLO detects a dangerous class.
2. Detection pauses.
3. FEEDOFF is sent.
4. The app waits 3 seconds.
5. FANON is sent.
6. The fan runs for 20 seconds.
7. FANOFF is sent.
8. Detection resumes.
9. The app waits 5 seconds.
10. FEEDON is sent.

The app includes guards to prevent overlapping clear sequences and avoids unsafe feeder/fan actions when Arduino disconnects.


11. Output and Configuration
----------------------------

Configuration file:
- config.ini

Example saved settings:
- camera_index
- monitor_camera_index
- monitor_camera_exposure
- resolution_text
- fps_limit_text
- camera_exposure
- yolo_model_path
- yolo_enabled
- yolo_class_filter
- yolo_dangerous_classes
- Arduino settings

Output folders:
- output/captures_detected
- output/captures_original


12. Production Safety Notes
---------------------------

- Test cameras and Arduino before production.
- If using two USB cameras, avoid using the same USB hub when possible.
- If detection becomes slow, reduce detection camera resolution or FPS first.
- The monitor camera is not part of AI detection and should not replace the detection camera.
- Check exposure, focus, and lighting before production because they strongly affect accuracy.
- Tune thresholds using real production images.
- Before enabling Auto cleansing, manually test FEEDON, FEEDOFF, FANON, and FANOFF to confirm relay direction and hardware behavior.


13. Resolution List
-------------------

resolutions =

['Source/Native','2592x1944','2592x1440','2560x1440','2048x1536','2304x1296','1920x1080','1600x1200','1600x900','1280X960','1280x720','1024x768','960X720','1024x576','960x540','800x600','848x480','800x450','640x480','640x360']


14. Blurry Image Check Command
------------------------------

Use this command to check dataset images before training or labeling:

    python.exe check_blurry_png.py "\\pc-max\D\docker\label-studio\media\upload\9" --threshold 100 --csv blur_report.csv


คู่มือภาษาไทย
=============

ไฟล์หลักของแอป: onefileV15.py

แอปนี้เป็นโปรแกรม GUI สำหรับตรวจจับความผิดปกติของเมล็ด/วัตถุจากภาพกล้องแบบ live โดยใช้ PyTorch, OpenCV, PyQt6 และสามารถเชื่อมต่อ Arduino เพื่อสั่งอุปกรณ์หน้างาน เช่น servo, feeder และ fan ได้


1. แอปนี้ทำอะไรได้บ้าง
----------------------

1. ตรวจจับความผิดปกติจากกล้อง live
   - เปิดกล้องหลักสำหรับงานตรวจจับ
   - แสดงภาพ live พร้อมผลการตรวจจับ
   - แสดง FPS, ขนาดภาพ, จำนวน detection และค่า focus

2. รองรับหลายโหมดตรวจจับ
   - Reconstruction/Model: ใช้โมเดล autoencoder เพื่อตรวจจับภาพที่ผิดจากตัวอย่างปกติ
   - Color (HSV): ใช้ช่วงสี HSV เพื่อตรวจจับสีเป้าหมาย
   - Hybrid (OR): ใช้ model หรือ HSV อย่างใดอย่างหนึ่งเจอก็ถือว่า detection
   - YOLO Only: ใช้ YOLO model ตรวจจับ object/class โดยตรง

3. ปรับ HSV ได้หลายช่วง
   - Hue1, Hue2, Hue3 สำหรับแยกช่วงสีหลายกลุ่ม
   - เปิด/ปิดแต่ละ Hue ได้
   - คลิกบนภาพเพื่อ sample สีจากภาพจริง
   - มี Hue target lock และ center marker ช่วยตั้งค่าหน้างาน

4. ใช้ YOLO เพิ่มเติมได้
   - เลือก YOLO model (.pt/.onnx)
   - เปิด/ปิด YOLO detection
   - ปรับ YOLO confidence
   - กรอง class ที่ต้องการตรวจจับ
   - ตั้ง dangerous class เพื่อให้ระบบ pause/สั่ง feeder/fan ได้

5. มีกล้อง Monitor Live View แยก
   - ใช้กล้องอีกตัวเพื่อดูสภาพภายในห้องเครื่อง/เครื่องจักร
   - กล้อง monitor เป็น live view อย่างเดียว ไม่เข้า model, YOLO, HSV, counter หรือ Arduino
   - ตั้งค่าคงที่: 1280x720, 30fps, MJPG
   - ปรับ Monitor Exposure แยกจากกล้องตรวจจับได้
   - ป้องกันไม่ให้ใช้กล้องตัวเดียวกับกล้อง detection พร้อมกัน

6. เชื่อมต่อ Arduino ได้
   - เลือก serial port และ baudrate
   - ส่ง trigger/clear command สำหรับ servo
   - ตั้ง trigger delay และ clear delay
   - สั่ง Feeder ON/OFF
   - สั่ง Fan ON/OFF
   - มีสถานะ Arduino และสถานะ servo/trigger บนหน้าจอ

7. ระบบหยุดและเคลียร์อัตโนมัติ
   - Stop feeder on detection: เมื่อเจอ anomaly ส่ง FEEDOFF และหยุด detection
   - Service breaker: ถ้าเจอ detection ต่อเนื่องหลายครั้งจะ pause detection เพื่อป้องกันเครื่องทำงานผิดปกติซ้ำ
   - Auto cleansing: เมื่อเจอ dangerous YOLO class จะหยุด detection, ปิด feeder, เปิด fan เคลียร์, resume detection และเปิด feeder กลับตามลำดับ

8. บันทึกผลและภาพ
   - Auto-save detections
   - Capture Image ระหว่าง live detection
   - บันทึกภาพ original และภาพ processed ลง output folder
   - แสดง popup/summary เมื่อมี detection หรือหยุดงาน

9. ใช้เทรน/เตรียมโมเดลต่อได้
   - มีปุ่ม Train Model สำหรับเรียก pipeline training ที่อยู่ในโปรเจกต์
   - มีเครื่องมือ check_blurry_png.py สำหรับตรวจภาพเบลอก่อนนำไป train หรือ label


2. ประโยชน์ของแอป
------------------

- ช่วยตรวจจับเมล็ดหรือวัตถุผิดปกติจากกล้องแบบ real-time
- ลดการตรวจด้วยสายตา และช่วยให้การคัดแยกสม่ำเสมอขึ้น
- ใช้ร่วมกับ Arduino เพื่อสั่งกลไกจริง เช่น feeder, servo หรือ fan
- มีกล้อง monitor แยกสำหรับดูสภาพเครื่องจักรโดยไม่ทำให้ระบบ AI ตรวจจับหนักขึ้น
- ปรับ threshold, HSV, YOLO confidence และ Arduino delay ได้จาก UI โดยไม่ต้องแก้ code
- เก็บภาพ detection ไว้ตรวจย้อนหลังหรือใช้ปรับปรุง dataset/model ต่อได้


3. อุปกรณ์และซอฟต์แวร์ที่ใช้
-----------------------------

อุปกรณ์ที่แนะนำ:
- คอมพิวเตอร์ Windows
- GPU NVIDIA ถ้าต้องใช้ model/YOLO บน GPU
- กล้อง USB สำหรับ detection 1 ตัว
- กล้อง USB สำหรับ monitor live view 1 ตัว ถ้าต้องการ
- Arduino สำหรับควบคุม servo/feeder/fan ถ้าใช้งาน hardware

Python package หลัก:
- PyQt6
- opencv-python
- numpy
- torch
- torchvision
- ultralytics ถ้าใช้ YOLO
- pyserial ถ้าใช้ Arduino


4. วิธีเปิดโปรแกรม
-------------------

เปิด terminal ในโฟลเดอร์โปรเจกต์ แล้วรัน:

    python onefileV15.py

หรือถ้าใช้ virtual environment ให้ activate venv ก่อน แล้วค่อยรันไฟล์นี้


5. วิธีใช้งานพื้นฐาน
--------------------

1. เลือกโหมดตรวจจับ
   - Reconstruction/Model: ต้อง Browse model ก่อน
   - Color (HSV): ใช้ได้โดยไม่ต้อง load model
   - Hybrid (OR): ใช้ model ร่วมกับ HSV
   - YOLO Only: ต้อง Browse YOLO model ก่อน

2. เลือกกล้อง detection
   - กด List Cameras
   - เลือก Source
   - เลือก Backend: Auto, MSMF หรือ DSHOW
   - เลือก FourCC, Frame Rate, Resolution
   - ปรับ Exposure ของกล้อง detection

3. ตั้งค่า threshold
   - MSE Threshold สำหรับ model/reconstruction
   - CV Threshold และ Contour Area สำหรับ contour/detection filtering
   - YOLO Confidence สำหรับ YOLO

4. กด Start Detection
   - แอปจะเปิดกล้องหลัก
   - ส่ง frame เข้า DetectionWorker
   - แสดงผลใน Live Feed

5. ระหว่างทำงาน
   - Pause Detection เพื่อหยุดชั่วคราว
   - Resume Detection เพื่อทำต่อ
   - Stop Detection เพื่อหยุดงาน
   - Capture Image เพื่อบันทึกภาพขณะ live


6. วิธีใช้ Monitor Live View
-----------------------------

Monitor camera ใช้สำหรับดูภาพห้องเครื่อง/สภาพเครื่องจักรเท่านั้น ไม่เกี่ยวกับ AI detection

วิธีใช้:
1. กด List Cameras
2. เลือก Monitor Camera
3. ปรับ Monitor Exposure ถ้าภาพสว่าง/มืดเกินไป
4. กด Start Monitor
5. จะมีหน้าต่าง Monitor Live View แยกขึ้นมา
6. กด Stop Monitor หรือปิดหน้าต่าง monitor เพื่อหยุดกล้อง

ค่าคงที่ของ Monitor camera:
- Resolution: 1280x720
- Frame rate: 30fps
- FourCC: MJPG
- Exposure: ปรับได้จาก Monitor Exposure slider

หมายเหตุ:
- กล้อง monitor ต้องไม่เป็นตัวเดียวกับกล้อง detection ขณะที่ detection กำลังทำงาน
- ถ้ากล้องกระตุก ควรเสียบกล้อง detection และ monitor แยก USB bus/port ให้ชัดเจน


7. วิธีใช้ HSV Detection
-------------------------

1. เลือกโหมด Color (HSV) หรือ Hybrid (OR)
2. เปิด Hue1/Hue2/Hue3 ตามสีที่ต้องการตรวจ
3. ปรับ Hue Low/High, Saturation Min, Value Min
4. ใช้ click sample บนภาพเพื่อช่วยเลือกสีจากภาพจริง
5. ใช้ Test Image ทดสอบกับภาพนิ่งก่อนใช้กับ live camera ได้

เหมาะกับงานที่ defect มีสีเด่น เช่น จุดดำ สีผิดปกติ หรือวัตถุสีแปลกจากพื้นหลัง


8. วิธีใช้ YOLO
----------------

1. กด Browse YOLO...
2. เลือกไฟล์ .pt หรือ .onnx
3. เปิด Enable YOLO detection
4. ปรับ YOLO Confidence
5. ใส่ YOLO Class Filter ถ้าต้องการตรวจเฉพาะบาง class
6. ใส่ YOLO Dangerous Classes ถ้าต้องการให้ class บางชนิด trigger การหยุด/เคลียร์อัตโนมัติ

ตัวอย่าง Dangerous Classes:

    Black_Head, 2

ใส่ได้ทั้งชื่อ class หรือ class id ตาม model ที่ใช้


9. วิธีใช้ Arduino
-------------------

1. ต่อ Arduino กับคอมพิวเตอร์
2. เลือก Port และ Baudrate
3. กด Connect
4. ตั้งค่า:
   - Enable trigger on anomaly
   - Trigger Delay
   - Auto-clear
   - Clear Delay
   - Trigger Command
   - Clear Command
5. ใช้ Test Trigger / Send Clear ทดสอบก่อนเริ่มงานจริง
6. ใช้ Feeder ON/OFF และ Fan ON/OFF เพื่อทดสอบ relay/manual control

Command ที่ใช้งานกับ Arduino ในระบบนี้:
- FEEDON
- FEEDOFF
- FANON
- FANOFF
- Trigger command เช่น 1
- Clear command เช่น 0


10. Auto Cleansing / Dangerous Class
------------------------------------

เมื่อเปิด Auto cleansing และ Arduino connected:

1. YOLO เจอ dangerous class
2. Detection ถูก pause
3. ส่ง FEEDOFF
4. รอ 3 วินาที
5. ส่ง FANON
6. เปิด fan เคลียร์ 20 วินาที
7. ส่ง FANOFF
8. Resume detection
9. รอ 5 วินาที
10. ส่ง FEEDON

ระบบมี guard ป้องกัน sequence ซ้อนกัน และถ้า Arduino disconnect ระหว่าง sequence จะไม่สั่ง feeder/fan ต่อแบบสุ่ม


11. Output และไฟล์ config
--------------------------

ไฟล์ตั้งค่า:
- config.ini

ตัวอย่างค่าที่บันทึก:
- camera_index
- monitor_camera_index
- monitor_camera_exposure
- resolution_text
- fps_limit_text
- camera_exposure
- yolo_model_path
- yolo_enabled
- yolo_class_filter
- yolo_dangerous_classes
- arduino settings

โฟลเดอร์ output:
- output/captures_detected
- output/captures_original


12. ข้อควรระวังในการใช้งานจริง
-------------------------------

- ทดสอบกล้องและ Arduino ทุกครั้งก่อนเริ่ม production
- ถ้าใช้กล้อง 2 ตัว ควรเสียบแยก USB controller หรืออย่างน้อยไม่ใช้ hub เดียวกัน
- ถ้า detection ช้า ให้ลด resolution/FPS ของกล้อง detection ก่อน
- Monitor camera ไม่เข้า AI จึงไม่ควรใช้แทนกล้อง detection
- ตรวจสอบ exposure/focus/แสงก่อนเริ่มงาน เพราะมีผลต่อความแม่นยำมาก
- ตั้ง threshold ด้วยภาพจริงจากหน้างาน ไม่ควรใช้ค่า default โดยไม่ทดสอบ
- ก่อนเปิด Auto cleansing ควรทดสอบ FEEDON/FEEDOFF/FANON/FANOFF แบบ manual ให้แน่ใจว่า relay ทำงานถูกทิศทาง


13. รายการ Resolution ที่มีใน UI
---------------------------------

resolutions =

['Source/Native','2592x1944','2592x1440','2560x1440','2048x1536','2304x1296','1920x1080','1600x1200','1600x900','1280X960','1280x720','1024x768','960X720','1024x576','960x540','800x600','848x480','800x450','640x480','640x360']


14. คำสั่งตรวจภาพเบลอ
----------------------

ใช้ตรวจภาพใน dataset ก่อนนำไป train/label:

    python.exe check_blurry_png.py "\\pc-max\D\docker\label-studio\media\upload\9" --threshold 100 --csv blur_report.csv
