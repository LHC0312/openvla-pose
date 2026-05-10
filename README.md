# OpenVLA-Pose

OpenVLA-OFT + LIBERO 재현 + cross-embodiment 연구용 작업 공간.

## 현재 단계: 공식 baseline 재현

연구 본격화 전에 **OpenVLA-OFT를 LIBERO(MuJoCo)에서 직접 굴리는 것**부터.

| 모델 | `moojink/openvla-7b-oft-finetuned-libero-spatial` (HF) |
|---|---|
| Sim | LIBERO (MuJoCo) — spatial / object / goal / 10 |
| 공식 코드 | https://github.com/moojink/openvla-oft, https://github.com/Lifelong-Robot-Learning/LIBERO |

---

## 사용법

### 1. 로컬 — LIBERO sim viewer (Mac)

LIBERO 환경을 MuJoCo viewer로 직접 봅니다. OpenVLA 추론은 안 함 (M4에선 불가).

```bash
bash scripts/setup_libero_local.sh                       # 1회 (conda env libero-sim 생성)
conda activate libero-sim
python local/libero_view.py --list                       # task 목록
python local/libero_view.py --suite libero_spatial --task 0
python local/libero_view.py --suite libero_object --task 0 --action wave
```

Viewer 마우스: 좌-드래그 회전 / 우-드래그 이동 / 스크롤 줌. macOS에서 viewer 안정성 이슈 시 `mjpython` 사용.

### 2. Colab — OpenVLA-OFT + LIBERO rollout

A100 권장 (16GB VRAM). 한 task 5 trial 기준 ~10분.

👉 https://colab.research.google.com/github/LHC0312/openvla-pose/blob/main/notebooks/00_libero_oft_reproduce.ipynb

노트북이 자동으로:
- `moojink/openvla-oft` + `Lifelong-Robot-Learning/LIBERO` clone
- pretrained 체크포인트 다운 (~15GB → Drive)
- sample inference + LIBERO MuJoCo rollout
- 영상 mp4로 Drive 저장

---

## 디렉토리 구조

```
openvla-pose/
├── notebooks/
│   └── 00_libero_oft_reproduce.ipynb     # Colab 메인
├── local/
│   └── libero_view.py                    # LIBERO MuJoCo viewer (Mac)
├── scripts/
│   └── setup_libero_local.sh             # 로컬 conda env + LIBERO 설치
├── external/                             # (gitignored)
│   └── LIBERO/                           # 공식 LIBERO 레포 clone
└── README.md
```

연구의 GNN/cross-embodiment 부분은 **baseline 재현이 끝난 뒤** 별도 디렉토리에 추가합니다.
