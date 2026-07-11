""" score model checkpoint """ 
from rollout import load_policy, run_policy_episode
from pathlib import Path
import torch
import numpy as np
import argparse
from sim import SimEnv
from expert import Phase


def score_ckpt(ckpt_path: str,  
            seed: int,
            max_steps: int = 1400, 
            episodes: int = 100,
            expert_baseline: bool = False) -> dict[str, float | list[float]]:
    
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
    parser.add_argument("--expert-baseline", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    score_dict = score_ckpt(
        args.ckpt_path,
        args.seed,
        args.max_steps,
        args.episodes,
        args.expert_baseline,
    )
    print(f"Mean Score: {score_dict['mean_score']}")
    print(f"Scores: {score_dict['scores']}")
    
if __name__ == "__main__":
    main()
