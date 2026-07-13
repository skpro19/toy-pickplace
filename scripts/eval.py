""" score model checkpoint """ 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from rollout import load_policy, make_episode_seeds, run_policy_episode
from pathlib import Path
import torch
import numpy as np
import argparse
from sim import SimEnv
from expert import Phase


_eval_worker_state = None


def initialize_eval_worker(ckpt_path: str) -> None:
    global _eval_worker_state

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    model, normalize, action_space, norm_dict = load_policy(
        model_path=Path(ckpt_path),
        device=device,
    )
    sim = SimEnv(randomize_scene=True)
    _eval_worker_state = (
        sim,
        model,
        device,
        normalize,
        action_space,
        norm_dict,
    )


def score_episode_worker(task: tuple[int, int, bool]) -> float:
    if _eval_worker_state is None:
        raise RuntimeError("evaluation worker was not initialized")

    seed, max_steps, expert_baseline = task
    sim, model, device, normalize, action_space, norm_dict = _eval_worker_state
    sim.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)
    last_phase = Phase.MOVE_ABOVE_CUBE

    def last_phase_cb(step, phase) -> None:
        del step
        nonlocal last_phase
        last_phase = phase

    with torch.inference_mode():
        run_policy_episode(
            sim=sim,
            model=model,
            device=device,
            normalize=normalize,
            action_space=action_space,
            norm_dict=norm_dict,
            max_steps=max_steps,
            track_phase=True,
            dagger=expert_baseline,
            beta=1.0 if expert_baseline else 0.0,
            rng=rng,
            phase_callback=last_phase_cb,
            should_stop=lambda: last_phase == Phase.DONE,
        )

    phases = list(Phase)
    return phases.index(last_phase) / phases.index(Phase.DONE)


def score_ckpt(ckpt_path: str,  
            seed: int,
            max_steps: int, 
            episodes: int,
            expert_baseline: bool = False,
            workers: int = 1) -> dict[str, float | list[float]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    if workers > 1:
        episode_seeds = make_episode_seeds(seed=seed, episodes=episodes)
        tasks = [
            (episode_seed, max_steps, expert_baseline)
            for episode_seed in episode_seeds
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, episodes),
            mp_context=context,
            initializer=initialize_eval_worker,
            initargs=(ckpt_path,),
        ) as executor:
            scores = list(executor.map(score_episode_worker, tasks))
        return {
            "scores": scores,
            "mean_score": float(np.mean(scores)),
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, norm_dict = load_policy(model_path=Path(ckpt_path), device=device)
    
    sim = SimEnv(randomize_scene=True, seed=seed)
    rng = np.random.default_rng(seed)

    store_dict = {
        "scores": [],
        "mean_score": 0.0,
    }

    phases = list(Phase)
    done_phase_index = phases.index(Phase.DONE)

    with torch.no_grad():
        for _ in range(episodes):
            last_phase = Phase.MOVE_ABOVE_CUBE

            def last_phase_cb(step, phase) -> None:
                nonlocal last_phase
                last_phase = phase

            run_policy_episode(sim=sim,
                             model=model,
                             device=device,
                             norm_dict=norm_dict,
                             max_steps=max_steps,
                             track_phase = True,
                             dagger=expert_baseline,
                             beta=1.0 if expert_baseline else 0.0,
                             rng = rng,
                            phase_callback = last_phase_cb,
                            should_stop=lambda: last_phase == Phase.DONE,
                            )

            phase_index = phases.index(last_phase)
            store_dict["scores"].append(phase_index / done_phase_index)

    store_dict["mean_score"] = float(np.mean(store_dict["scores"]))
    return store_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=1400)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expert-baseline", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    score_dict = score_ckpt(
        args.ckpt_path,
        args.seed,
        args.max_steps,
        args.episodes,
        args.expert_baseline,
        args.workers,
    )
    print(f"Mean Score: {score_dict['mean_score']}")
    print(f"Scores: {score_dict['scores']}")
    
if __name__ == "__main__":
    main()
