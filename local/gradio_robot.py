"""Gradio app — Robot GNN Explorer.

Validates: "Can different-DOF robot arms be represented as a graph,
and can a single latent z transfer pose semantics across them?"

Run:
    cd local && python gradio_robot.py
Opens http://localhost:7860 in browser.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gradio as gr

from robot_graph import ROBOTS, fk_chain, graph_to_pyg_edges
from viz_3d import plot_two
from gnn_min import MeanGNNEncoder, ChainDecoder


# Single shared encoder (untrained — just for shape/forward validation in stage 1)
torch.manual_seed(0)
ENCODER = MeanGNNEncoder(z_dim=16)
DECODER = ChainDecoder(z_dim=16)
MAX_SLIDERS = 8


@torch.inference_mode()
def explore(teacher_name, student_name, *teacher_angles_padded):
    teacher = ROBOTS[teacher_name]()
    student = ROBOTS[student_name]()

    # Use only the first teacher.n_joints sliders
    t_angles = torch.tensor(teacher_angles_padded[:teacher.n_joints], dtype=torch.float32)

    # 1. Teacher FK
    t_pos = fk_chain(teacher, t_angles)             # [N_t+1, 3]
    t_edges = graph_to_pyg_edges(teacher)            # [2, E_t]

    # 2. Encode teacher pose graph -> z
    z = ENCODER(t_pos, t_edges)

    # 3. Decode z to student joint angles (different DOF!)
    s_angles = DECODER(z, n_joints=student.n_joints)

    # 4. Student FK
    s_pos = fk_chain(student, s_angles)
    s_edges = graph_to_pyg_edges(student)

    # 5. Plot side-by-side
    fig = plot_two(
        t_pos, t_edges, f'TEACHER: {teacher.name}',
        s_pos, s_edges, f'STUDENT: {student.name}',
    )

    info = (
        f'z[:8] = {[f"{v:+.2f}" for v in z[:8].tolist()]}\n'
        f'Teacher angles ({teacher.n_joints}): {[f"{a:+.2f}" for a in t_angles.tolist()]}\n'
        f'Student angles ({student.n_joints}): {[f"{a:+.2f}" for a in s_angles.tolist()]}\n'
        f'Teacher graph: {teacher.n_joints+1} nodes, {t_edges.size(1)//2} undirected edges\n'
        f'Student graph: {student.n_joints+1} nodes, {s_edges.size(1)//2} undirected edges'
    )
    return fig, info


def build_app():
    with gr.Blocks(title='Robot GNN Explorer') as app:
        gr.Markdown(
            '# Robot GNN Explorer — Stage 1 Validation\n'
            '**Question**: 임의의 DOF를 가진 로봇 팔이 GNN으로 표현 가능하고, '
            '같은 잠재 z로 서로 다른 로봇이 의미적으로 같은 포즈를 만들 수 있는가?\n\n'
            '- Teacher: 슬라이더로 joint 조정 → forward kinematics → 3D 그래프\n'
            '- Encoder: pose graph → z (16-dim, 노드 수 무관)\n'
            '- Decoder: z + student의 DOF → student joint angles\n'
            '- Student: forward kinematics → 3D 그래프\n\n'
            '**주의**: GNN은 untrained. 1단계는 *그래프 표현/시각화/cross-DOF forward*가 동작하는지 검증.'
        )

        with gr.Row():
            teacher_dd = gr.Dropdown(
                choices=list(ROBOTS.keys()),
                value='Franka-like 7-DOF', label='Teacher robot',
            )
            student_dd = gr.Dropdown(
                choices=list(ROBOTS.keys()),
                value='UR5-like 6-DOF', label='Student robot',
            )

        sliders = [
            gr.Slider(-3.14, 3.14, value=0.3 * (i % 2 - 0.5), step=0.05, label=f'Teacher J{i}')
            for i in range(MAX_SLIDERS)
        ]
        plot_out = gr.Plot(label='Robot graphs (Teacher vs Student)')
        info_out = gr.Textbox(label='z + angles + graph stats', lines=6)

        all_inputs = [teacher_dd, student_dd] + sliders
        for inp in all_inputs:
            inp.change(explore, all_inputs, [plot_out, info_out])

        app.load(explore, all_inputs, [plot_out, info_out])

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(server_name='127.0.0.1', server_port=7860, share=False, inbrowser=True)
