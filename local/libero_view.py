"""LIBERO task viewer — opens MuJoCo GLFW window on macOS.

Run from libero-sim conda env:
    conda activate libero-sim
    python local/libero_view.py                       # default: spatial task 0
    python local/libero_view.py --suite libero_object --task 3
    python local/libero_view.py --list                # list all tasks

Mouse: left-drag rotate, right-drag pan, scroll zoom.
ESC or close window to quit.
"""
import argparse
import os
import sys

from libero.libero import benchmark, get_libero_path


SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90']


def list_tasks(suite_name=None):
    bdict = benchmark.get_benchmark_dict()
    suites = [suite_name] if suite_name else SUITES
    for s in suites:
        if s not in bdict:
            continue
        b = bdict[s]()
        print(f'\n{s}  ({b.n_tasks} tasks):')
        for i in range(b.n_tasks):
            t = b.get_task(i)
            print(f'  [{i:2d}] {t.language}')


def view(suite='libero_spatial', task_idx=0, max_steps=2000, action_mode='zero'):
    # Lazy import to keep --list fast
    import numpy as np
    import mujoco
    import mujoco.viewer
    from libero.libero.envs.env_wrapper import ControlEnv

    bdict = benchmark.get_benchmark_dict()
    if suite not in bdict:
        print(f'Unknown suite: {suite}. Choose from {SUITES}')
        sys.exit(1)
    b = bdict[suite]()
    if task_idx >= b.n_tasks:
        print(f'Task index {task_idx} out of range (suite has {b.n_tasks})')
        sys.exit(1)

    task = b.get_task(task_idx)
    bddl_path = os.path.join(
        get_libero_path('bddl_files'), task.problem_folder, task.bddl_file
    )
    print(f'Suite: {suite}')
    print(f'Task : [{task_idx}] {task.language}')
    print(f'BDDL : {bddl_path}')
    print(f'Action mode: {action_mode}')

    # NOTE: has_renderer=False because we use mujoco.viewer directly.
    # robosuite's onscreen renderer is unreliable on macOS arm64 + mujoco 3.x.
    env = ControlEnv(
        bddl_file_name=bddl_path,
        robots=['Panda'],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
    )
    env.reset()

    # Pull the raw mujoco MjModel / MjData from robosuite's wrapper
    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data

    print('\nViewer open. Mouse: left=rotate, right=pan, scroll=zoom. Close window or Ctrl+C to exit.')

    rng = np.random.default_rng(0)
    def get_action():
        if action_mode == 'zero':
            return np.zeros(7)
        if action_mode == 'random':
            a = rng.uniform(-0.3, 0.3, size=7)
            a[6] = rng.choice([-1.0, 1.0])  # gripper open/close
            return a
        if action_mode == 'wave':
            # Smooth EE rotation back and forth
            t = get_action.t
            get_action.t += 1
            a = np.zeros(7)
            a[3] = 0.4 * np.sin(t * 0.05)  # roll
            a[4] = 0.4 * np.cos(t * 0.05)  # pitch
            return a
        return np.zeros(7)
    get_action.t = 0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            while viewer.is_running() and step < max_steps:
                env.step(get_action())
                viewer.sync()
                step += 1
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='libero_spatial', choices=SUITES)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--list', action='store_true', help='List all tasks and exit')
    ap.add_argument('--max-steps', type=int, default=10000)
    ap.add_argument('--action', choices=['zero', 'random', 'wave'], default='zero',
                    help='zero=정지, random=흔들기, wave=EE 회전')
    args = ap.parse_args()

    if args.list:
        list_tasks()
        return

    view(args.suite, args.task, args.max_steps, args.action)


if __name__ == '__main__':
    main()
