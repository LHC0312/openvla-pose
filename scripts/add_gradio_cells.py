"""Append Gradio demo cells to notebooks 01 and 04."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NB01 = ROOT / 'notebooks' / '01_setup_and_inference.ipynb'
NB04 = ROOT / 'notebooks' / '04_gnn_pose_space.ipynb'


def md(text):
    return {
        'cell_type': 'markdown', 'metadata': {},
        'source': text.splitlines(keepends=True) if isinstance(text, str) else text,
    }


def code(src):
    return {
        'cell_type': 'code', 'metadata': {},
        'execution_count': None, 'outputs': [],
        'source': src.splitlines(keepends=True) if isinstance(src, str) else src,
    }


# ---------- Notebook 01: OpenVLA inference demo ----------
nb01_md = """## 8. Gradio 데모 — 웹에서 인터랙티브 테스트

`share=True`로 실행하면 https://xxxx.gradio.live 형태 공개 URL 발급.
- 핸드폰/다른 PC에서도 접속 가능
- 72시간 동안 유효 (Colab 세션이 살아있는 동안)
"""

nb01_code = """!pip install -q gradio

import gradio as gr
import matplotlib.pyplot as plt

def vla_predict(image, instruction):
    if image is None or not instruction:
        return 'Need image + instruction', None, ''
    img = Image.fromarray(image).convert('RGB').resize((224, 224))
    prompt = f'In: What action should the robot take to {instruction}?\\nOut:'
    inputs = processor(prompt, img).to('cuda', dtype=torch.bfloat16)
    try:
        action = vla.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)
        action_str = (f'dx={action[0]:+.3f}  dy={action[1]:+.3f}  dz={action[2]:+.3f}\\n'
                      f'drx={action[3]:+.3f}  dry={action[4]:+.3f}  drz={action[5]:+.3f}\\n'
                      f'gripper={action[6]:+.3f}')
    except Exception as e:
        action_str = f'predict_action failed: {e}'
    h = extract_hidden_state(img, instruction)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.bar(range(min(64, h.shape[0])), h[:64].numpy())
    ax.set_title(f'Hidden state (first 64 of {h.shape[0]})')
    ax.set_xlabel('dim'); ax.set_ylabel('value')
    plt.tight_layout()
    stats = f'norm={h.norm():.3f}  mean={h.mean():+.3f}  std={h.std():.3f}'
    return action_str, fig, stats

demo = gr.Interface(
    fn=vla_predict,
    inputs=[
        gr.Image(label='Image'),
        gr.Textbox(label='Instruction', value='pick up the red cup'),
    ],
    outputs=[
        gr.Textbox(label='7-DOF action', lines=4),
        gr.Plot(label='Hidden state'),
        gr.Textbox(label='Stats'),
    ],
    title='OpenVLA Inference + Hidden State',
    description='Image + instruction → 7-DOF action + LLM hidden state',
)
demo.launch(share=True, debug=False)
"""

# ---------- Notebook 04: GNN pose space explorer ----------
nb04_md = """## 7. Gradio 데모 — z 슬라이더로 잠재 포즈 공간 탐색

각 z 차원을 슬라이더로 조절 → decoder가 즉시 robot pose 그림.
"이 차원이 무슨 의미인지" 직관적으로 보는 도구.
"""

nb04_code = """!pip install -q gradio

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

Z_DIM = 32
N_SLIDERS = 8  # 처음 8차원만 슬라이더로

def explore(*z_vals):
    z = torch.zeros(Z_DIM)
    for i, v in enumerate(z_vals):
        z[i] = v
    decoder.eval()
    with torch.no_grad():
        angles = decoder(z)
    pos = fake_fk_chain(angles)  # [N, 3]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    p = pos.numpy()
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=50, c='C0')
    for i in range(len(p) - 1):
        ax.plot(p[i:i+2, 0], p[i:i+2, 1], p[i:i+2, 2], 'k-', lw=2)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-0.5, 0.5)
    ax.set_title(f'Robot pose from z[:{N_SLIDERS}]')
    angle_str = '  '.join(f'{a:+.2f}' for a in angles.numpy())
    return fig, f'angles: {angle_str}'

inputs = [gr.Slider(-2.0, 2.0, value=0.0, label=f'z[{i}]') for i in range(N_SLIDERS)]
outputs = [gr.Plot(label='Decoded skeleton'), gr.Textbox(label='Joint angles')]

demo = gr.Interface(
    fn=explore,
    inputs=inputs, outputs=outputs,
    title='GNN Pose Space Explorer',
    description='Move sliders → see decoded robot pose. Other z dims are 0.',
    live=True,
)
demo.launch(share=True, debug=False)
"""


def append_cells(path, new_cells):
    nb = json.loads(path.read_text())
    nb['cells'].extend(new_cells)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f'{path.name}: now {len(nb["cells"])} cells')


append_cells(NB01, [md(nb01_md), code(nb01_code)])
append_cells(NB04, [md(nb04_md), code(nb04_code)])
print('Done.')
