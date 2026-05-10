# OpenVLA-Pose: SE(3) Equivariant GNN Pose Space for Cross-Embodiment VLA

## 개요

이 프로젝트는 OpenVLA를 기반으로, 관절 각도 대신 **SE(3) invariant 잠재 포즈 공간**을 통해
cross-embodiment robot manipulation을 달성하는 연구 코드베이스입니다.

## 환경 가정

- **로컬 (M4 Pro Mac)**: 노트북 편집, 시각화, 포즈 데이터 처리
- **Colab (T4/A100)**: OpenVLA 가중치 다운로드, 추론, LoRA 파인튜닝, GNN 학습

CUDA가 없는 로컬에서는 OpenVLA 7B 추론조차 매우 느리므로, 모든 모델 무거운 작업은 Colab에서 수행합니다.

## 디렉토리 구조

```
openvla-pose/
├── notebooks/          # Colab에서 실행하는 .ipynb
│   ├── 01_setup_and_inference.ipynb     # 환경 세팅 + 가중치 다운로드 + 추론 smoke test
│   ├── 02_lora_finetune.ipynb           # OpenVLA LoRA 파인튜닝
│   ├── 03_pose_extraction.ipynb         # 영상에서 3D 관절 추출 → GNN 학습 데이터
│   └── 04_gnn_pose_space.ipynb          # SE(3) Equivariant GNN encoder/decoder 학습
├── local/              # 로컬 시각화/분석 도구
│   ├── viz_skeleton.py                  # 3D 스켈레톤 뷰어 (matplotlib)
│   ├── viz_latent.py                    # 잠재 공간 t-SNE/UMAP
│   └── check_equivariance.py            # equivariance 검증 스크립트
├── scripts/            # 보조 스크립트
│   ├── setup_local_env.sh               # 로컬 venv 세팅
│   └── drive_sync.py                    # Colab Drive ↔ 로컬 동기화
├── configs/            # 학습 설정 YAML
├── data/               # (gitignored) 데이터셋
├── checkpoints/        # (gitignored) 모델 가중치
└── docs/               # 설계 문서
```

## 워크플로우

### Step 1. 로컬 환경 세팅 (1회)

```bash
cd "/Users/haechan/robot(5.10)/openvla-pose"
bash scripts/setup_local_env.sh
```

이 스크립트는:
- Python 3.10 venv를 `local/.venv`에 생성
- PyTorch (MPS), torch-geometric, e3nn, jupyter, plotly 등 설치
- CUDA 버전 X — 로컬은 시각화/검증 전용

### Step 2. Colab에서 OpenVLA 추론 (notebook 01)

1. Google Drive 마운트
2. `openvla/openvla-7b` 가중치 다운로드 (~15GB) → Drive에 저장
3. BridgeData 예시 이미지로 추론 smoke test
4. **hidden state 추출 함수 검증** ← 우리 연구의 핵심 입력

산출물: `data/openvla_hidden_states.pt` (Drive)

### Step 3. Colab에서 LoRA 파인튜닝 (notebook 02)

A100 권장. T4도 가능하지만 batch_size 1.

### Step 4. 포즈 데이터 추출 (notebook 03)

- MediaPipe (가벼움, 즉시 가능) 또는 HaMeR (정확도 ↑)
- 영상 → 3D 관절 시퀀스 → numpy 저장

### Step 5. GNN 학습 (notebook 04)

- e3nn 기반 SE(3) equivariant encoder
- (z, joint_angles) 페어로 robot decoder 학습
- 산출물: GNN encoder/decoder 가중치

### Step 6. 로컬에서 시각화/검증

```bash
source local/.venv/bin/activate
python local/viz_skeleton.py --pose data/sample_pose.npy
python local/check_equivariance.py --encoder checkpoints/gnn_encoder.pt
```

## 중요한 용어 정정

원래 제안서의 "SE(3) Equivariant"는 정확히는 **"SE(3) Invariant latent + SE(3) Equivariant decoder"** 입니다.
- Invariant: 회전해도 z가 같음 (encoder)
- Equivariant: z의 변환이 출력의 변환과 commutative (decoder)

## 결과 보는 방법 (Drive 다운만이 답이 아님)

| 방법 | 시나리오 |
|---|---|
| **셀 inline 출력** | matplotlib/plotly — Colab UI에 즉시 표시 |
| **Gradio share URL** | 노트북 01/04 마지막 셀. `demo.launch(share=True)` → `https://xxxx.gradio.live` 발급, 핸드폰에서도 접속 |
| **W&B (wandb.ai)** | 학습 메트릭 실시간 클라우드 대시보드 (학생 무료) |
| **TensorBoard** | Colab 내 `%tensorboard --logdir runs` |
| **Drive sync + 로컬 viz** | 결과 `.npz` Drive 저장 → 로컬 `local/viz_*.py`로 분석 |

## 위험 신호 (먼저 검증할 것)

1. **OpenVLA hidden state alignment**: 사람/로봇 영상의 hidden state가 실제로 가까운지 → notebook 01에서 검증
2. **MediaPipe 3D pose 정확도**: 실제 EPIC-KITCHENS 영상에서 합리적인지 → notebook 03에서 검증
3. **Equivariance 깨짐**: e3nn 구현이 실제로 invariant인지 → `check_equivariance.py`로 매 학습마다 자동 검증
