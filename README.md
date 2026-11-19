# TesaModelTest

ชุดรันโมเดลแยกเมล็ดกาแฟบน Raspberry Pi ใช้ YOLO26n-seg 640×640 แบบ ONNX FP32 จากรอบ B3 และ confidence 0.35 ภายในมีโมเดล ภาพ Test 32 ภาพ และเฉลย COCO รายเมล็ด 369 ชิ้น ครอบคลุม 7 คลาส ได้แก่ black, broken, good, husk, immature, parchment และ shell

## ติดตั้งบน Raspberry Pi

ใช้ Raspberry Pi OS 64-bit พร้อม Python 3.10 ขึ้นไปและ pip หลัง clone หรือคัดลอกโฟลเดอร์ ให้เปิด Terminal ในไดเรกทอรีที่มี README นี้ แล้วรันทีละบรรทัด

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

ติดตั้งครั้งแรกต้องใช้อินเทอร์เน็ต รุ่นนี้ใช้ Ultralytics สำหรับเตรียมภาพและแปลงผล mask จึงมี torch และ torchvision เป็น dependency แม้โมเดลจะรันด้วย ONNX Runtime CPU น้ำหนักอยู่ใน weights/ แล้ว

## ทดสอบความแม่นยำและเวลาบน Pi

```bash
python run_test.py
```

คำสั่งนี้เรียก infer.py ในโปรเซสใหม่ จับเวลาตั้งแต่เปิดโปรเซสจนเขียนไฟล์ครบ แล้วหารด้วยจำนวนภาพที่ประมวลผล ค่า end-to-end รวมการโหลดโมเดล และจบการจับเวลาก่อนอ่านเฉลยเพื่อประเมินผล ปิด warm-up สำหรับการจับเวลารอบนี้

- `outputs_test/metrics.json` เก็บค่าเฉลี่ยทั้งชุด คะแนนรายภาพ และ end_to_end_seconds_per_image
- `outputs_test/seg_mask/` เก็บ PNG รายภาพที่มี instance ID ตรงกับ predictions.json
- `outputs_test/predictions.json` เก็บ ID คลาส และ polygon ของแต่ละเมล็ด
- `outputs_test/vis/` เก็บ overlay สำหรับตรวจภาพ
- `outputs_test/benchmark.json` เก็บ median ภายในโปรเซส ใช้ end_to_end_seconds_per_image ใน metrics.json เมื่อต้องการเวลาตามวิธีกรรมการ

## เทียบทุกโมเดลบน Pi

`models/` มี **37 ชุดรัน** รวม 29 แบบที่ผ่านการ optimize และไฟล์ ONNX จาก checkpoint ของ 7 รอบเทรนหลัก พร้อม A0 ที่ใช้ classifier อีกแบบ

| ชื่อในคำสั่ง | ที่มา |
|---|---|
| A0, A2, A3 | U-Net 512 baseline, U-Net 512 Copy-Paste และ U-Net 640 |
| A1_cls | U-Net A0 กับ classifier ทางเลือกที่เห็นพื้นหลังในกรอบตัด |
| B0, B1, B2, B3 | YOLO11n-seg 512, YOLO26n-seg 512, YOLO26n-seg 640 และรุ่น 640 Copy-Paste |
| unet_* | U-Net หลายขนาดและรุ่น INT8, pruning, distillation, QAT |
| yolo_* | YOLO26n-seg รุ่นที่คัดไว้ หลายขนาดและ runtime FP32, INT8, NCNN |

```bash
python benchmark_all.py --list
python benchmark_all.py --split valid
python benchmark_all.py --split valid --model A0 B0 B3 yolo_640 yolo_640_ncnn yolo_640_int8
```

ค่าเริ่มต้นใช้ Valid 33 ภาพเพื่อเลือกรุ่นและจูน รันทีละโมเดลด้วยภาพและจำนวน threads ที่กำหนดเท่ากัน แล้วเขียน `outputs_all_valid/comparison.csv` หลังจบแต่ละโมเดล คอลัมน์ accuracy ใน CSV อยู่ในช่วง 0–1 และคอลัมน์ `end_to_end_seconds_per_image` เป็นวินาที/ภาพจากเครื่องที่รัน ผลรายโมเดลมี `metrics.json` พร้อม F1 รายคลาสและ confusion matrix, `infer.log`, PNG, JSON และ overlay ถ้ารุ่นใดล้มเหลว สคริปต์จะบันทึกสถานะ error และเดินต่อจนจบคิว

ถ้าจะจูน inference ให้คัดลอกโฟลเดอร์รุ่นใต้ `models/` เป็นชื่อใหม่ แล้วแก้ `settings.conf` สำหรับ YOLO หรือค่าใน `watershed` สำหรับ U-Net ที่ `manifest.json` ใช้ `--model` ระบุทั้งรุ่นเดิมและรุ่นที่ปรับ เพื่อเทียบผลบน Valid โดยตรง

เมื่อเลือกโมเดลและค่าตั้งแล้ว ให้แทน `yolo_640` ด้วยชื่อโฟลเดอร์รุ่นที่เลือกเพื่อประเมิน Test ที่กันไว้

```bash
python benchmark_all.py --split test --model yolo_640
```

ใช้ `python benchmark_all.py --split test` เมื่อต้องการเทียบทั้ง 37 รุ่นบน Test หากนำผลชุดนี้ไปเลือกหรือปรับโมเดล ให้ใช้ชุดภาพรอบใหม่สำหรับยืนยันผลสุดท้าย

`checkpoints/` มี `best.pt` ของรอบ A0/A2/A3 และ B0–B3 ตัว classifier ทั้งคู่ และ checkpoint จากการทดลองบีบอัด U-Net เพื่อใช้ต่อบนเครื่องเทรน สำหรับ YOLO ใช้ `checkpoints/B3/best.pt` เป็นตัวอย่างจุดเริ่ม fine-tune พร้อมชี้ `data=` ไปยัง YAML ของชุดฝึก โฟลเดอร์นี้แนบเฉพาะ Valid และ Test สำหรับ benchmark ส่วน U-Net ใช้สคริปต์เทรนเดิมร่วมกับ checkpoint และชุด Train

## รันเฉพาะ inference กับภาพของตัวเอง

```bash
python infer.py --input test --weights weights --out outputs_custom
```

เปลี่ยน test เป็นโฟลเดอร์ที่มีไฟล์ JPG, JPEG หรือ PNG อยู่ชั้นแรก ชื่อภาพก่อนนามสกุลต้องไม่ซ้ำกัน ผลทำนายสร้างได้จากภาพอย่างเดียว ส่วน run_test.py ใช้เฉลย test/_annotations.coco.json ของชุดที่แนบมา

## ผลอ้างอิงจากการรันแพ็กเกจบน Windows CPU

| ตัววัด | Test 32 ภาพ |
|---|---:|
| Instance IoU | 74.43% |
| Counting แบบจับคู่ | 82.34% |
| ทายคลาสถูกในคู่ที่ IoU ≥ 0.50 | 67.64% |
| Class-aware F1 ที่ IoU ≥ 0.50 | 60.84% |
| คลาดเคลื่อนด้านจำนวนเฉลี่ย | 2.09 เมล็ด/ภาพ |
| เวลารวมทั้งคำสั่งหารจำนวนภาพ บน Windows | 0.231 วินาที/ภาพ |

การจับคู่ใช้การจับคู่หนึ่งต่อหนึ่งที่ให้ IoU รวมสูงสุด Counting นับคู่ที่ทับกันจริง หารด้วยจำนวนที่มากกว่าระหว่าง GT กับ prediction แล้วเฉลี่ยรายภาพ ข้อความ mask repaired หรือ omitted ระหว่างรันมาจากขั้นตอนสร้าง polygon ให้สอดคล้องกับ PNG คะแนนด้านบนวัดจากไฟล์หลังผ่านขั้นตอนนี้แล้ว

ทดสอบคำสั่งครบชุดบน Windows โดยใช้ runtime ที่ติดตั้งอยู่ในเครื่อง เวลาบน Pi และการติดตั้งบน Pi เครื่องใหม่ยังต้องตรวจบนบอร์ดจริง
