import torch
import torchvision
from pathlib import Path


def get_project_root():
    p = Path.cwd()
    while p.name != "zoo_vision":
        if p == p.parent:
            raise RuntimeError(
                "Could not find a path named zoo_vision in the hierarchy. Cannot determine project root."
            )
        p = p.parent
    return p


PROJECT_ROOT = get_project_root()


def compile_model(weights_path: Path, output_path: Path) -> None:
    print(f"Compiling {weights_path}")
    print("Loading empty model...")
    if weights_path.name.startswith("maskrcnn_c2_"):
        model = torchvision.models.get_model(
            "maskrcnn_resnet50_fpn_v2",
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )
    else:
        raise RuntimeError("Unknown model")

    print("Loading weights from disk...")
    checkpoint = torch.load(PROJECT_ROOT / weights_path, weights_only=False)

    print("Restoring weights...")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    traced_module = torch.jit.script(model)
    traced_module.save(output_path)


def main() -> None:
    weights_paths = PROJECT_ROOT.glob("models/**/*.pth")
    for weights_path in weights_paths:
        output_path = PROJECT_ROOT / weights_path.with_suffix(".ptc")
        if not output_path.exists():
            compile_model(weights_path, output_path)


if __name__ == "__main__":
    main()
