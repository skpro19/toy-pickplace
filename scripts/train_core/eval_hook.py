from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from eval import ScoreResult, score_ckpt


def evaluate_checkpoint(
    *,
    model_path: Path,
    writer: SummaryWriter,
    epoch: int,
    seed: int,
    episodes: int,
    max_steps: int,
    workers: int,
) -> ScoreResult:
    score_dict = score_ckpt(
        ckpt_path=str(model_path),
        seed=seed,
        max_steps=max_steps,
        episodes=episodes,
        workers=workers,
    )
    mean_score = float(score_dict["mean_score"])
    max_score = max(score_dict["scores"])
    writer.add_scalar("Eval/mean_score", mean_score, epoch)
    writer.add_scalar("Eval/max_score", max_score, epoch)
    writer.add_scalar("Eval/grasp_rate", score_dict["grasp_rate"], epoch)
    writer.add_scalar("Eval/lift_rate", score_dict["lift_rate"], epoch)
    writer.add_scalar("Eval/tray_reach_rate", score_dict["tray_reach_rate"], epoch)
    writer.add_scalar(
        "Eval/lowered_to_tray_rate",
        score_dict["lowered_to_tray_rate"],
        epoch,
    )
    writer.add_scalar(
        "Eval/released_over_tray_rate",
        score_dict["released_over_tray_rate"],
        epoch,
    )
    writer.add_scalar(
        "Eval/placement_success_rate",
        score_dict["placement_success_rate"],
        epoch,
    )
    writer.flush()
    return score_dict
