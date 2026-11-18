# Fine-tuning SAM3 cho phân đoạn ngữ nghĩa (VOC 2012, Cityscapes, PASCAL-Context 59)

Repo này fine-tune SAM3 (`facebookresearch/sam3`, commit `f1647e8`) trên tập train của VOC 2012 và Cityscapes,
và (job riêng, mục 4) PASCAL-Context 59.
Có hai cấu hình:
- **A2:** train phần sau backbone, khoảng 32.6M tham số;
- **A1:** chỉ train `segmentation_head`, khoảng 2.3M tham số.

Bốn model chạy lần lượt trong **một lệnh**, theo thứ tự A2 Cityscapes → A2 VOC → A1 VOC → A1 Cityscapes.

---

## 1. Cài đặt

```bash
git clone <url-repo-này> sam3_finetune
cd sam3_finetune
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# torch cho CUDA 13.0 (driver 580)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# SAM3 (ghim commit) + các thư viện còn lại
pip install -r requirements.txt
```

## 2. Dữ liệu và checkpoint

Tải theo link ở các mục dưới, giải nén, rồi đặt vào thư mục `data/` và `weights/` trong repo đúng như các cây thư mục
(hai thư mục này đã có trong `.gitignore`). Đặt ở chỗ khác cũng được, khi đó sửa đường dẫn trong `paths.yaml` (mục 2.4).

### 2.1 VOC 2012

Link tải: https://drive.google.com/file/d/10Dw_dc4xiMh6AxcABGJXvVLOyRv82M4H/view?usp=sharing

```
data/VOC2012/                    <- voc_root
├── JPEGImages/
├── SegmentationClass/
├── SegmentationObject/
└── ImageSets/Segmentation/      train.txt (1464 ảnh), val.txt (1449 ảnh)
```

### 2.2 Cityscapes

Link tải: https://drive.google.com/file/d/15l2ej2XgN0Pxe5AaO7Jl7ftF9sdBMjTy/view?usp=sharing

```
data/data_cityscapes/
├── leftImg8bit_trainvaltest/leftImg8bit/    <- cityscapes_leftImg8bit   (train/<city>/..., val/<city>/...)
└── gtFine_trainvaltest/gtFine/              <- cityscapes_gtFine        (cần cả *_labelIds.png và *_instanceIds.png)
```

### 2.3 Checkpoint SAM3

Link tải `sam3.pt`: https://drive.google.com/file/d/1I6XixMFkaSCdHvTh_z4kljd62-2DBoyy/view?usp=sharing
Đặt vào `weights/sam3.pt`.

### 2.4 `paths.yaml`

```bash
cp paths.example.yaml paths.yaml
```

`paths.example.yaml` đã trỏ sẵn tới đúng bố cục ở trên (`data/`, `weights/`, output ở `outputs/`). Nếu dữ liệu nằm chỗ khác,
sửa các khóa `voc_root`, `cityscapes_leftImg8bit`, `cityscapes_gtFine`, `sam3_checkpoint`, `out_root`.

## 3. Chạy

```bash
python run_finetune.py --config paths.yaml
```

Lệnh này chạy toàn bộ:
1. kiểm tra môi trường và dữ liệu;
2. chuẩn bị dữ liệu train;
3. preflight: kiểm tra SAM3 và chạy thử train trong khoảng 30–45 phút; lỗi thì dừng trước khi train thật;
4. train và đánh giá lần lượt 4 model;
5. đóng gói kết quả.

Log được ghi vào `outputs/logs/`.

**Bị ngắt giữa chừng** (hết giờ, bị kill, mất điện): **chạy lại đúng lệnh trên.**
- Các bước đã xong được bỏ qua.
- Model đang train tiếp tục từ cuối epoch gần nhất đã lưu.
- Bước đánh giá val tiếp tục từ ảnh đang dở.

## 4. PASCAL-Context 59 (job riêng)

PC-59 chạy bằng **lệnh riêng**, độc lập với lệnh ở mục 3; chạy trước hay sau đều được. Job này chạy SAM3 gốc trên
val, rồi train 2 model (A2 rồi A1); output ghi vào `outputs/pc59/`.

### 4.1 Dữ liệu

Link tải: https://drive.google.com/drive/folders/1dCwcG-Pyh_FwatlaPVYjBrcNs2yUwGWk?usp=sharing
(`VOC2010.zip`, `SegmentationClassContext.zip`, `pascal_context_train.txt`, `pascal_context_val.txt`; `59_labels.txt`
không cần)

```
data/VOC2010/                      <- pc59_root (từ VOC2010.zip)
├── JPEGImages/                    ảnh <id>.jpg
├── SegmentationClassContext/      các file <id>.png từ SegmentationClassContext.zip (10103 file)
├── pascal_context_train.txt
└── pascal_context_val.txt
```

Các thư mục khác trong VOC2010 (`Annotations/`, `ImageSets/`, ...) không dùng. Nếu server đã có `full_pipeline/pc59/data/`
của pipeline cũ (cùng 4 mục trên) thì không cần tải, chỉ cần trỏ `pc59_root` tới đó.

### 4.2 Chạy

```bash
python run_finetune_pc59.py --config paths.yaml
```

Các bước:
1. kiểm tra, chuẩn bị dữ liệu, preflight (chạy thử train trên PC-59);
2. **SAM3 gốc** (chưa fine-tune) trên 5105 ảnh val ở ngưỡng 0.5;
3. train và đánh giá A2, rồi A1;
4. đóng gói.

**Bị ngắt thì chạy lại đúng lệnh trên**; bước đánh giá val tiếp tục từ ảnh đang dở. Job này cần khoảng 10 GB ổ
trống (PC-59 không lưu `val_raw/`); log ở `outputs/pc59/logs/`.

```
outputs/pc59/
├── logs/, preflight/, data_coco/
├── pc59_base/               SAM3 gốc: val_report.json, val_conf_tau.npz, val_pred/ (ảnh dự đoán ở 0.5), val_conf/
├── pc59_A2/, pc59_A1/       best.pt, best.json, holdout_history.json, val_report.json, val_conf_tau.npz,
│                            val_pred/ (ảnh dự đoán ở ngưỡng chọn trên holdout), val_conf/, ...
├── sam3ft_pc59_summary.csv  bảng kết quả: base, A2, A1
└── sam3ft_pc59_results_pack.zip
```

## 5. Cấu trúc thư mục

```
sam3_finetune/
├── run_finetune.py          lệnh chính, VOC + Cityscapes (chạy lại thì tự tiếp tục)
├── run_finetune_pc59.py     lệnh riêng cho PC-59
├── ft_common.py             hằng số, nhãn, mIoU, tiện ích
├── ft_prepare_data.py       GT -> COCO + RLE
├── ft_model.py              nạp sam3.pt, đóng băng theo A1/A2, lưu / nạp best.pt
├── ft_trainer.py            Trainer của SAM3 + đánh giá holdout mỗi epoch, early stopping
├── ft_train.py              train một model
├── ft_eval.py               suy luận và đánh giá
├── configs/ft_sam3_seg.yaml cấu hình train (từ template của repo SAM3)
├── assets/                  chia fit / holdout, dự đoán tham chiếu cho preflight
├── tools/make_split.py      tạo / kiểm tra file chia fit / holdout
├── requirements.txt, paths.example.yaml
├── data/, weights/          dữ liệu và sam3.pt (không đưa lên git)
└── outputs/
    ├── logs/                log
    ├── <ds>_<cfg>/          mỗi model: best.pt, best.json, holdout_history.json, val_report.json, val_raw/, ...
    ├── sam3ft_summary.csv   bảng kết quả, mỗi model một dòng
    ├── sam3ft_results_pack.zip
    └── pc59/                job PC-59 (mục 4.2)
```

## 6. Gửi lại kết quả

- **`outputs/sam3ft_results_pack.zip`**: các file nhỏ (json, csv, log, confusion); đây là file chính.
- Nếu được thì gửi thêm `outputs/<ds>_<cfg>/<ds>_<cfg>_best.pt` của 4 model (A2 khoảng 130 MB, A1 khoảng 10 MB)
  và các thư mục `*_val_raw/` (mỗi model khoảng 1–2 GB, nên nén riêng).
- Job PC-59:
  - **`outputs/pc59/sam3ft_pc59_results_pack.zip`** (file chính);
  - 3 thư mục ảnh dự đoán `pc59_base/pc59_base_val_pred/`, `pc59_A2/pc59_A2_val_pred/`, `pc59_A1/pc59_A1_val_pred/`
    (khoảng 30 MB mỗi thư mục, nén chung một file);
  - nếu được thì thêm `pc59_A2_best.pt`, `pc59_A1_best.pt`.

## 7. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `checks failed: sam3 commit ...` | `pip install --force-reinstall "sam3[train] @ git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66"` |
| `sha256 of sam3.pt ...` | sai file checkpoint, tải lại `sam3.pt` theo link ở mục 2.3 |
| `found N images, expected ...` | đường dẫn dataset trong `paths.yaml` trỏ sai |
| `missing keys ['pc59_root']` | thêm `pc59_root` vào `paths.yaml` (mục 4.1) |
| `GT files 'inst' missing` | thiếu `*_gtFine_instanceIds.png` trong `gtFine` |
| `cannot import triton` | cài torch bản CUDA cho Linux (đã kèm triton), hoặc `pip install triton` |
| `numpy ... incompatible` / lỗi `cv2` | `pip install "numpy<2" opencv-python-headless==4.10.0.84` |
| bị kill `-9` (hết RAM) | giảm `num_workers` trong `paths.yaml` (ví dụ 2), rồi chạy lại đúng lệnh |
| `... was made with other settings` | thư mục output cũ được tạo với cấu hình khác; đổi tên thư mục đó rồi chạy lại |
| `preflight failed ...` | gửi lại `outputs/preflight/preflight_report.json` (PC-59: `outputs/pc59/preflight/`) và log |

Khi báo lỗi, gửi kèm file log trong `outputs/logs/` (PC-59: `outputs/pc59/logs/`).
