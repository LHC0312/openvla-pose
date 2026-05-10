# OpenVLA-Pose

OpenVLA-OFT + LIBERO 재현 + cross-embodiment 연구용 작업 공간.

## 현재 단계: 공식 baseline 재현

| 모델 | `moojink/openvla-7b-oft-finetuned-libero-spatial` (HF) |
|---|---|
| Sim | LIBERO (MuJoCo) — spatial / object / goal / 10 |
| 공식 코드 | https://github.com/moojink/openvla-oft, https://github.com/Lifelong-Robot-Learning/LIBERO |

---

## 1. 로컬 셋업 (Mac M4, conda env `libero-sim`)

원샷 셋업:
```bash
bash scripts/setup_libero_local.sh    # LIBERO sim + viewer (1회)
bash scripts/setup_openvla_local.sh   # OpenVLA-OFT 추가 설치 (1회)
```

### 1a. LIBERO sim viewer
```bash
conda activate libero-sim
python local/libero_view.py --list
python local/libero_view.py --suite libero_spatial --task 0
python local/libero_view.py --suite libero_object --task 0 --action wave
```
Viewer 마우스: 좌-드래그 회전 / 우-드래그 이동 / 스크롤 줌.

### 1b. OpenVLA-OFT 로컬 추론 (sample observation)
```bash
conda activate libero-sim
python local/openvla_inference.py --device mps
```
- 첫 실행 시 가중치 ~15GB 다운로드 (`~/.cache/huggingface/`)
- M4 MPS로 한 step inference: 추정 분 단위 (CPU는 더 느림)
- 추론 자체보다 "코드/모델 구조 직접 확인" 용도

`HF_HOME=/external/path` 로 가중치 위치 변경 가능.

---

## 2. Colab — 정식 LIBERO rollout (A100 권장)

A100에서 한 task 5 trial 기준 ~10분.

👉 https://colab.research.google.com/github/LHC0312/openvla-pose/blob/main/notebooks/00_libero_oft_reproduce.ipynb

노트북이 자동으로:
- 공식 레포 clone
- 가중치 ~15GB → Drive
- sample inference + LIBERO MuJoCo rollout
- 영상 mp4 저장

---

## 디렉토리 구조

```
openvla-pose/
├── notebooks/
│   └── 00_libero_oft_reproduce.ipynb    # Colab 메인
├── local/
│   ├── libero_view.py                    # MuJoCo viewer (LIBERO Franka)
│   └── openvla_inference.py              # OpenVLA-OFT 로컬 추론
├── scripts/
│   ├── setup_libero_local.sh             # LIBERO 설치
│   └── setup_openvla_local.sh            # OpenVLA-OFT 설치
├── external/                             # (gitignored)
│   ├── LIBERO/                           # 공식 LIBERO
│   └── openvla-oft/                      # 공식 OpenVLA-OFT
└── README.md
```

GNN/cross-embodiment 부분은 baseline 재현 끝난 뒤 별도 모듈로 추가.
