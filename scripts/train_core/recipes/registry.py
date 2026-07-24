from train_core.recipes.mlp import MlpRecipe
from train_core.recipes.vision_mlp import VisionMlpRecipe
from train_core.types import TrainingRecipe


def get_recipe(*, arch: str) -> TrainingRecipe:
    if arch == MlpRecipe.architecture:
        return MlpRecipe()
    if arch == VisionMlpRecipe.architecture:
        return VisionMlpRecipe()
    raise ValueError(f"Unsupported training recipe architecture: {arch!r}")
