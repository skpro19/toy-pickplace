from sim import SimEnv
from pathlib import Path

SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"

def main():
    sim = SimEnv(scene_path=SCENE_PATH)
    m, d = sim.model, sim.data

    # print(f"type(model): {type(model)}")
    # print(f"dir(model): {dir(model)}")
    # print(f"type(data): {type(data)}")

    for name in sorted(dir(d)):
        if name.startswith("_"):
            continue
        val = getattr(d, name, None)
        if hasattr(val, "shape"):
            print(f"{name:20} {val.shape}")

    num_episodes = 10 

    for episode in range(num_episodes):
        sim.reset_episode()
        

if __name__ == "__main__":
    main()

