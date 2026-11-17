# Fine-tuning SAM3 cho phân đoạn ngữ nghĩa (VOC 2012, Cityscapes)

Repo này fine-tune SAM3 (`facebookresearch/sam3`, commit `f1647e8`) trên tập train của VOC 2012 và Cityscapes.
Có hai cấu hình:

| Cấu hình | Phần được train | Tham số train | Phần đóng băng |
|---|---|---|---|
| **A2** | encoder + decoder của transformer, `dot_prod_scoring`, `segmentation_head`, neck của backbone ảnh (`convs`), `resizer` của nhánh text | khoảng 32.6M | ViT ảnh (447M), text encoder (353M), geometry encoder |
| **A1** | chỉ `segmentation_head` | khoảng 2.3M | mọi phần còn lại |

Thứ tự chạy cố định: **A2 Cityscapes → A2 VOC → A1 VOC → A1 Cityscapes**. A1 xếp cuối hàng đợi: nếu hết thời gian thì
A2 vẫn đã có đủ kết quả. Cả bốn model chạy **lần lượt trong một job**, chỉ bằng **một lệnh**.

---

## 1. Yêu cầu

- GPU có bf16 và ít nhất 30 GB bộ nhớ. Code được viết cho H200 MIG `2g.35gb`; batch 1 ảnh 1008×1008.
- RAM từ 32 GB trở lên. `num_workers: 4` (template của repo dùng 10, dễ bị kill `-9` khi hết RAM).
- Ổ đĩa trống **≥ 25 GB** tại `out_root`.
- Python 3.10–3.12, driver NVIDIA phù hợp với bản CUDA của torch.

## 2. Cài đặt

```bash
git clone <url-repo-này> sam3_finetune
cd sam3_finetune
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# (a) torch trước, đúng bản CUDA của máy. Driver 580 / CUDA 13.0: dùng bản cu130
#     (trùng bản đã chạy các lượt SAM3 gốc trên Kaggle: torch 2.11.0)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# (b) SAM3 ghim commit + các thư viện còn lại
pip install -r requirements.txt
```

Bước 0 của pipeline tự kiểm tra lại các điều kiện sau:
- import đủ thư viện;
- GPU có bf16;
- sam3 đúng commit `f1647e8d0c1d280991f4d1b7d974c2704537fe66`;
- sha256 của `sam3.pt`;
- số ảnh của từng dataset;
- dung lượng đĩa còn trống.

## 3. Dữ liệu và checkpoint

### 3.1 VOC 2012 (`VOCtrainval_11-May-2012.tar`, **không** dùng SBD)

```
VOC2012/                         <- voc_root
├── JPEGImages/                  2007_000032.jpg ...
├── SegmentationClass/           PNG palette: 0 = nền, 1..20 = lớp, 255 = void (viền)
├── SegmentationObject/          PNG palette: id của từng vật thể (để tách instance)
└── ImageSets/Segmentation/
    ├── train.txt                1464 ảnh
    └── val.txt                  1449 ảnh
```

### 3.2 Cityscapes (`leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`)

```
leftImg8bit/                     <- cityscapes_leftImg8bit
├── train/<city>/<city>_xxxxxx_yyyyyy_leftImg8bit.png      2975 ảnh
└── val/<city>/...                                          500 ảnh
gtFine/                          <- cityscapes_gtFine
├── train/<city>/<city>_xxxxxx_yyyyyy_gtFine_labelIds.png
│                <city>_xxxxxx_yyyyyy_gtFine_instanceIds.png   <- BẮT BUỘC (tách từng người / xe)
└── val/<city>/..._gtFine_labelIds.png
```

Hai thư mục `leftImg8bit/` và `gtFine/` đặt ở đâu cũng được, vì đường dẫn của từng thư mục được khai báo riêng.
Nếu thiếu `*_instanceIds.png`, bước 0 sẽ dừng và báo lỗi. Khi đó cần chép **toàn bộ** thư mục `gtFine` từ
`gtFine_trainvaltest.zip`.

### 3.3 Checkpoint

`sam3.pt` là checkpoint ảnh chính thức của SAM3 (sha256 `9999e234…c8c9e`). Bước 0 kiểm tra sha256 này
(chỉ tính lại khi file thay đổi).

### 3.4 `paths.yaml`

```bash
cp paths.example.yaml paths.yaml      # rồi sửa đường dẫn
```

| Khóa | Ý nghĩa |
|---|---|
| `voc_root` | thư mục `VOC2012/` ở mục 3.1 |
| `cityscapes_leftImg8bit`, `cityscapes_gtFine` | hai thư mục ở mục 3.2 |
| `sam3_checkpoint` | file `sam3.pt` |
| `out_root` | nơi ghi mọi output (tự tạo nếu chưa có) |
| `device` | `cuda` khi chạy trên server (`cpu` chỉ để thử code trên máy cá nhân) |
| `num_workers` | số worker của dataloader, mặc định 4 |

Đường dẫn tương đối được tính từ vị trí của file `paths.yaml`. `paths.yaml` đã có trong `.gitignore`.

## 4. Chạy

```bash
# 1) xem kế hoạch: bước nào xong, bước nào còn phải chạy (không chạy gì)
python run_finetune.py --config paths.yaml --dry-run

# 2) (khuyên dùng, khoảng 30-60 phút, phần lớn là bước 1; COCO của bước 1 được dùng lại cho lần chạy thật)
#    chạy toàn bộ chuỗi trên vài ảnh (4 ảnh train, 2 epoch, 2 ảnh val mỗi model), output ở <out_root>_smoke
python run_finetune.py --config paths.yaml --smoke

# 3) chạy thật: toàn bộ từ đầu đến cuối
python run_finetune.py --config paths.yaml
```

Job dài nên chạy bằng `nohup` hoặc SLURM. Log luôn được ghi vào `<out_root>/logs/run_<thời gian>.log`.

```bash
nohup python run_finetune.py --config paths.yaml > run.out 2>&1 &
```

```bash
#!/bin/bash
#SBATCH --job-name=sam3ft
#SBATCH --gres=gpu:1            # sửa theo cấu hình MIG / partition của server
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<giới hạn của server>
cd /đường/dẫn/sam3_finetune && source .venv/bin/activate
python run_finetune.py --config paths.yaml
```

**Bị ngắt (hết giờ, kill `-9`, mất điện): chạy lại đúng lệnh cũ.**
- Các bước đã xong được bỏ qua, nhận biết qua fingerprint của cấu hình.
- Model đang train sẽ train tiếp từ cuối epoch gần nhất (`<ds>_<cfg>_resume/checkpoint.pt`).
- Bước đánh giá val chạy tiếp từ ảnh đang dừng.

Mỗi lần chạy lại chỉ mất phần công của epoch đang dở.

Muốn chạy riêng một số model:

```bash
python run_finetune.py --config paths.yaml --models cityscapes:A2 voc2012:A2
```

## 5. Pipeline làm gì

| Bước | Nội dung | File |
|---|---|---|
| 0 checks | thư viện, GPU, commit sam3, sha256 `sam3.pt`, số ảnh (VOC 1464/1449, City 2975/500), file GT, ổ đĩa | `run_finetune.py` |
| 1 prepare | GT của **train-fit** → COCO json + mask RLE, mỗi vật thể một annotation (xem 5.1) | `ft_prepare_data.py` |
| 2 preflight | (a) SAM3 gốc trên 50 ảnh holdout phải trùng ≥ 98% pixel với dự đoán của lượt chạy trên Kaggle; (b) chạy thử A2 và A1, 30 bước mỗi cấu hình, để đo tốc độ và bộ nhớ, rồi lưu và nạp lại `best.pt`; (c) ước tính số giờ | `run_finetune.py`, `ft_eval.py`, `ft_train.py` |
| 3–5 | với mỗi model: train trên fit → đánh giá holdout sau mỗi epoch → early stopping → đánh giá val **một lần** bằng checkpoint tốt nhất | `ft_train.py`, `ft_trainer.py`, `ft_eval.py` |
| 6 | `sam3ft_summary.csv` + `sam3ft_results_pack.zip` (gói được làm lại sau mỗi model) | `run_finetune.py` |

### 5.1 Dữ liệu train

- Tập train chính thức được chia cố định thành **fit (80%)** và **holdout (20%)**, đúng bằng cách chia của Phương pháp B
  (`assets/<ds>_split_train.json`): VOC 1171/293, City 2380/595.
- Train **chỉ dùng fit**. Holdout dùng để chọn epoch và ngưỡng. **Val chỉ dùng một lần, ở cuối.**
- Mỗi ảnh tạo ra một query text cho **mọi** lớp, gồm cả lớp không có trong ảnh (query âm, dạy presence head),
  đúng như `COCO_FROM_JSON(include_negatives=True)` của repo.
  - Tên prompt giống hệt lúc đánh giá; VOC tách từ: `dining table`, `potted plant`, `tv monitor`.
- Cách tách vật thể:
  - **VOC:** dùng `SegmentationObject`. Pixel void (255) cách vùng có nhãn ≤ 3 px nhận nhãn và vật thể của pixel có nhãn gần nhất.
    Void sâu hơn thì coi là nền.
  - **Cityscapes:**
    - person, rider, car, truck, bus, train, motorcycle, bicycle: mỗi instance (`instanceIds`) là một vật thể;
    - pole, traffic light, traffic sign: mỗi thành phần liên thông 8-hướng là một vật thể, mảnh nhỏ hơn 64 px gộp vào mảnh gần nhất;
    - các lớp stuff: một mask cho mỗi lớp;
    - pixel ignore (trainId 255) không thuộc vật thể nào.
- Augmentation, loss (box, presence, focal + dice cho mask, semantic) và matcher giữ nguyên template fine-tune của repo.

### 5.2 Train, early stopping, chọn ngưỡng

- Dùng `Trainer` chính thức của SAM3, không sửa:
  - AdamW, lr 8e-5 cho transformer / 2.5e-5 cho neck / 5e-6 cho resizer text;
  - lịch inverse-sqrt, clip 0.1, bf16, batch 1.
- `ft_trainer.FtTrainer` chỉ thay `run_val`. Sau mỗi epoch, model được đánh giá trên holdout bằng **đúng quy trình suy luận**
  của các lượt SAM3 gốc:
  - một prompt cho mỗi lớp;
  - score = sigmoid(logit) × sigmoid(presence);
  - mỗi pixel lấy lớp của mask có score cao nhất.
- mIoU được tính trên lưới ngưỡng 0.05…0.70. Ngưỡng τ được chọn theo quy tắc của Phương pháp B: mIoU cao nhất,
  hòa trong khoảng 0.001 thì lấy ngưỡng gần 0.5 nhất.
- Mỗi khi holdout mIoU vượt mức tốt nhất trước đó, chương trình ghi đè `<ds>_<cfg>_best.pt`. File này chỉ chứa phần trọng số
  được train (khoảng 130 MB với A2), không lưu cả model.
- Early stopping: dừng khi **2 epoch liên tiếp** không tăng quá **0.1 điểm mIoU**. Giới hạn trên là 12 epoch cho VOC
  và 6 epoch cho City, khoảng 14 nghìn bước mỗi dataset.
- Val dùng checkpoint tốt nhất và ngưỡng **τ_A chọn trên holdout**. Ngưỡng tốt nhất trên val chỉ được ghi lại, không dùng vào việc gì.

## 6. Cấu trúc thư mục

```
sam3_finetune/
├── run_finetune.py          lệnh duy nhất: bước 0 -> 6, chạy lại thì tự tiếp tục
├── ft_common.py             hằng số, nhãn, layout dataset, mIoU, quy tắc chọn tau, tiện ích I/O
├── ft_prepare_data.py       bước 1: GT -> COCO + RLE (train-fit)
├── ft_model.py              nạp sam3.pt (kiểm tra key), đóng băng theo A1/A2, lưu / nạp best.pt
├── ft_trainer.py            FtTrainer = Trainer của SAM3 + run_val (holdout, best.pt, early stop)
├── ft_train.py              train MỘT model (gọi single_proc_run của repo)
├── ft_eval.py               suy luận như các lượt SAM3 gốc; các mode val / preflight / reload / rawcheck
├── configs/ft_sam3_seg.yaml cấu hình Hydra (từ template roboflow_v100 của repo)
├── assets/
│   ├── <ds>_split_train.json       chia fit / holdout (giống Phương pháp B)
│   └── <ds>_ref_pred/              50 dự đoán tham chiếu từ Kaggle (preflight)
├── requirements.txt
├── paths.example.yaml
└── README.md
```

Output (`out_root`):

```
outputs/
├── checks.json                         kết quả bước 0
├── logs/run_<time>.log                 toàn bộ log
├── data_coco/
│   ├── <ds>_fit_coco.json              annotation COCO (RLE) của train-fit
│   ├── <ds>_fit_stats.json             thống kê: số vật thể / lớp, tỉ lệ void được lấp, ...
│   └── samples/<ds>_*.jpg              ảnh minh họa mask (để kiểm tra bằng mắt)
├── preflight/preflight_report.json     độ trùng với Kaggle, tốc độ, bộ nhớ, ước tính số giờ
├── <ds>_<cfg>/                         ví dụ cityscapes_A2/
│   ├── <ds>_<cfg>_best.pt              trọng số tốt nhất (chỉ phần được train)
│   ├── <ds>_<cfg>_best.json            epoch tốt nhất, tau_A, holdout mIoU, dừng sớm hay chạm trần
│   ├── <ds>_<cfg>_holdout_history.json mIoU holdout theo từng epoch và từng tau, thời gian, số bước
│   ├── <ds>_<cfg>_holdout_conf_tau.npz confusion holdout của epoch tốt nhất (mỗi ảnh, mỗi tau)
│   ├── <ds>_<cfg>_params.json          số tham số train / đóng băng theo nhóm
│   ├── <ds>_<cfg>_train_config.yaml    cấu hình Hydra đã resolve
│   ├── <ds>_<cfg>_train_log/           log của Trainer (train_stats.json, tensorboard)
│   ├── <ds>_<cfg>_train_done.json      đánh dấu train xong
│   ├── <ds>_<cfg>_val_report.json      mIoU val tại tau_A và tại 0.5, IoU từng lớp
│   ├── <ds>_<cfg>_val_conf_tau.npz     confusion val (mỗi ảnh, mỗi tau)
│   ├── <ds>_<cfg>_val_pred/<id>.png    dự đoán val tại tau_A
│   ├── <ds>_<cfg>_val_raw/<id>.npz     đầu ra thô (logit, mask, embedding ...) cùng định dạng các lượt SAM3 gốc
│   └── <ds>_<cfg>_done.json
├── sam3ft_summary.csv                  một dòng cho mỗi model
└── sam3ft_results_pack.zip             <- GỬI FILE NÀY VỀ
```

## 7. Gửi lại kết quả

Gửi lại hai file sau:
- **`<out_root>/sam3ft_results_pack.zip`**: các file nhỏ (json, csv, yaml, log, confusion npz, ảnh mẫu); không có `.pt` hay raw.
- Nếu được, thêm `<ds>_<cfg>_best.pt` của 4 model (khoảng 130 MB với A2, 10 MB với A1) và thư mục `*_val_raw/`.
  Thư mục raw dùng cho phân tích lỗi sau này, có thể nén riêng.

## 8. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `checks failed: sam3 commit ...` | `pip install --force-reinstall "sam3[train] @ git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66"` |
| `sha256 of sam3.pt ...` | sai file checkpoint (không phải `sam3.pt` chính thức) |
| `found N images, expected 1464` | `voc_root` trỏ sai, hoặc dùng bản VOC augmented (SBD); cần bản gốc |
| `GT files 'inst' missing` | thiếu `*_gtFine_instanceIds.png`, chép đủ thư mục `gtFine` |
| `cannot import triton` | focal loss của SAM3 là kernel Triton; cài torch bản CUDA cho Linux (đã kèm triton), hoặc `pip install triton` |
| `numpy ... incompatible` / lỗi `cv2` | `pip install "numpy<2" opencv-python-headless==4.10.0.84` |
| bị kill `-9` (hết RAM) | giảm `num_workers` trong `paths.yaml` (ví dụ 2), rồi chạy lại đúng lệnh |
| CUDA out of memory | báo lại kèm `preflight/preflight_report.json`; preflight in ra bộ nhớ đỉnh |
| `... was made with other settings` | thư mục output cũ được tạo với cấu hình khác; đổi tên thư mục đó rồi chạy lại |
| `preflight failed: only xx% pixels agree` | sai môi trường (commit / torch / checkpoint); gửi lại `preflight_report.json` |

Mọi lỗi đều có trong log `<out_root>/logs/`. Khi báo lỗi, gửi kèm file log này.
