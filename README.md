# Fine-tuning SAM3 cho phân đoạn ngữ nghĩa (VOC 2012, Cityscapes)

Repo này fine-tune SAM3 (`facebookresearch/sam3`, commit `f1647e8`) trên tập train của VOC 2012 và Cityscapes.
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

Tải về thư mục `data/` và `weights/` trong repo (hai thư mục này đã có trong `.gitignore`). Trên server không có trình duyệt
thì dùng `gdown`:

```bash
pip install gdown
mkdir -p data weights
gdown 10Dw_dc4xiMh6AxcABGJXvVLOyRv82M4H -O data/data_voc.zip         # VOC 2012
gdown 15l2ej2XgN0Pxe5AaO7Jl7ftF9sdBMjTy -O data/data_cityscapes.zip  # Cityscapes (khoảng 12 GB)
gdown 1I6XixMFkaSCdHvTh_z4kljd62-2DBoyy -O weights/sam3.pt           # SAM3
cd data && unzip -q data_voc.zip && unzip -q data_cityscapes.zip && rm data_voc.zip data_cityscapes.zip && cd ..
```

Nếu `gdown` báo lỗi quota của Google Drive, mở link ở các mục dưới bằng trình duyệt, tải về rồi chép lên server.

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

## 4. Cấu trúc thư mục

```
sam3_finetune/
├── run_finetune.py          lệnh duy nhất (chạy lại thì tự tiếp tục)
├── ft_common.py             hằng số, nhãn, mIoU, tiện ích
├── ft_prepare_data.py       GT -> COCO + RLE
├── ft_model.py              nạp sam3.pt, đóng băng theo A1/A2, lưu / nạp best.pt
├── ft_trainer.py            Trainer của SAM3 + đánh giá holdout mỗi epoch, early stopping
├── ft_train.py              train một model
├── ft_eval.py               suy luận và đánh giá
├── configs/ft_sam3_seg.yaml cấu hình train (từ template của repo SAM3)
├── assets/                  chia fit / holdout, dự đoán tham chiếu cho preflight
├── requirements.txt, paths.example.yaml
├── data/, weights/          dữ liệu và sam3.pt (không đưa lên git)
└── outputs/
    ├── logs/                log
    ├── <ds>_<cfg>/          mỗi model: best.pt, best.json, holdout_history.json, val_report.json, val_raw/, ...
    ├── sam3ft_summary.csv   bảng kết quả, mỗi model một dòng
    └── sam3ft_results_pack.zip
```

## 5. Gửi lại kết quả

- **`outputs/sam3ft_results_pack.zip`**: các file nhỏ (json, csv, log, confusion); đây là file chính.
- Nếu được thì gửi thêm `outputs/<ds>_<cfg>/<ds>_<cfg>_best.pt` của 4 model (A2 khoảng 130 MB, A1 khoảng 10 MB)
  và các thư mục `*_val_raw/` (mỗi model khoảng 1–2 GB, nên nén riêng).

## 6. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `checks failed: sam3 commit ...` | `pip install --force-reinstall "sam3[train] @ git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66"` |
| `sha256 of sam3.pt ...` | sai file checkpoint, tải lại `sam3.pt` theo link ở mục 2.3 |
| `found N images, expected ...` | đường dẫn dataset trong `paths.yaml` trỏ sai |
| `GT files 'inst' missing` | thiếu `*_gtFine_instanceIds.png` trong `gtFine` |
| `cannot import triton` | cài torch bản CUDA cho Linux (đã kèm triton), hoặc `pip install triton` |
| `numpy ... incompatible` / lỗi `cv2` | `pip install "numpy<2" opencv-python-headless==4.10.0.84` |
| bị kill `-9` (hết RAM) | giảm `num_workers` trong `paths.yaml` (ví dụ 2), rồi chạy lại đúng lệnh |
| `... was made with other settings` | thư mục output cũ được tạo với cấu hình khác; đổi tên thư mục đó rồi chạy lại |
| `preflight failed ...` | gửi lại `outputs/preflight/preflight_report.json` và log |

Khi báo lỗi, gửi kèm file log trong `outputs/logs/`.
