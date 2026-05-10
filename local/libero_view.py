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


def view(suite='libero_spatial', task_idx=0, max_steps=2000):
    # Lazy import to keep --list fast
    from libero.libero.envs import ControlEnv

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

    env = ControlEnv(
        bddl_file_name=bddl_path,
        robots=['Panda'],
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera='agentview',
        ignore_done=True,
    )
    env.reset()
    print('\nViewer open. Mouse to navigate. Ctrl+C to exit.')

    import numpy as np
    zero_action = np.zeros(7)
    try:
        for step in range(max_steps):
            env.step(zero_action)
            env.env.render()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='libero_spatial', choices=SUITES)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--list', action='store_true', help='List all tasks and exit')
    ap.add_argument('--max-steps', type=int, default=2000)
    args = ap.parse_args()

    if args.list:
        list_tasks()
        return

    view(args.suite, args.task, args.max_steps)


if __name__ == '__main__':
    main()
