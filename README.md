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
python -m pip install --isolated --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu'
python -m pip install -r requirements.txt 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu'
```

ติดตั้งครั้งแรกต้องใช้อินเทอร์เน็ต ติดตั้ง PyTorch และ torchvision จากดัชนี CPU ก่อน แล้วใช้เวอร์ชันเดียวกันตอนลง requirements เพื่อให้ pip เลือกชุด CPU สำหรับ Raspberry Pi รุ่นนี้ใช้ Ultralytics สำหรับเตรียมภาพและแปลงผล mask และใช้ ONNX Runtime CPU หรือ NCNN สำหรับ inference น้ำหนักอยู่ใน weights/ และ models/ แล้ว

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

### จูน 4 รุ่นตามเกณฑ์ Day 3

คำสั่งนี้รัน 17 ค่าเทียบกันบน Valid ได้แก่ U-Net A0 ที่ปรับ watershed 9 แบบ รวมค่าเดิม และ YOLO 640 FP32, NCNN และ INT8 ที่ลอง confidence รวม 8 แบบ

```bash
python benchmark_all.py --day3-tune
```

ค่าเริ่มต้นใช้ Valid 33 ภาพที่แนบมา หากมีชุดกองคละคลาส Day 3 ให้จัดเป็น COCO โดยวางภาพและ `_annotations.coco.json` ใน `valid/` ใต้รากชุดข้อมูล ชื่อคลาสใน COCO ให้ใช้ 7 ชื่อเดียวกับชุดแนบ เช่น `parchment` ใช้รายชื่อภาพที่กันจาก Train สำหรับเลือกค่าจูน

```bash
python benchmark_all.py --day3-tune --data-root /path/to/day3_coco --out outputs_day3_pi
```

`comparison.csv` เก็บคะแนนส่วน segmentation **เต็ม 30** จาก `15 × IoU + 15 × Counting` พร้อมเวลา median ต่อภาพ และเวลารวมคำสั่งเฉลี่ยต่อภาพ `ranking.json` จัดรุ่นที่ทั้งสองตัวเลขไม่เกิน 1 วินาทีไว้ก่อน แล้วเรียงตามคะแนนเต็ม 30 ค่านี้ใช้คัดรุ่นเพื่อซ้อมบน Pi ของกรรมการ ส่วนคะแนน MQTT ต้องยืนยันตอนซ้อมรับส่งจริง

เมื่อเลือกค่าจูนแล้ว ให้ใช้ `--split test` กับชุดทดสอบที่กันไว้และรันเฉพาะค่าที่เลือก ตัวอย่างสำหรับ U-Net ที่เลือก `boundary_045` และ YOLO INT8 ที่เลือก `0.175` โดยใช้ค่าจากผล Pi ของคุณแทนตัวอย่าง

```bash
python benchmark_all.py --split test --model A0 --watershed outputs_day3_tune/A0/boundary_045/watershed.json
python benchmark_all.py --split test --model yolo_640_int8 --confidence 0.175
```

### รายการรุ่นและคำสั่งทั่วไป

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

สำหรับ YOLO สามารถเทียบหลายค่า confidence จากโมเดลไฟล์เดียวกันได้ด้วยคำสั่งด้านล่าง ค่า 0.35 คือ baseline ของสองรุ่นนี้

```bash
python benchmark_all.py --split valid --model yolo_640_ncnn yolo_640_int8 \
  --confidence 0.20 0.25 0.30 0.35 0.40 0.45 --out outputs_tune_pi
```

ดูผลรวมที่ `outputs_tune_pi/comparison.csv` และคะแนนรายคลาสใน `metrics.json` ใต้โฟลเดอร์ของแต่ละรุ่นและค่า confidence ก่อนนำค่าที่เลือกไปยืนยันบน Test

จากการจูนบน Windows ค่าเด่นคือ `0.30` สำหรับ NCNN และ `0.175` สำหรับ INT8 เมื่อเน้น IoU กับ Counting เทียบกับค่าเดิม `0.35` ของแต่ละรุ่นบน Pi ด้วยคำสั่งนี้

```bash
python benchmark_all.py --split valid --model yolo_640_ncnn --confidence 0.30 0.35 --out outputs_tune_pi_ncnn
python benchmark_all.py --split valid --model yolo_640_int8 --confidence 0.15 0.175 0.20 0.35 --out outputs_tune_pi_int8
```

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
