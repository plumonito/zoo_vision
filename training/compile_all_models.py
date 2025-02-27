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


class ModelWithTransforms(torch.nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x):
        xn = self.transforms(x)
        y = self.model.forward(xn)
        return y


def compile_model(weights_path: Path, output_path: Path) -> None:
    print(f"Compiling {weights_path}")
    print("Loading empty model...")
    if weights_path.name.startswith("maskrcnn_c2_"):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )
        model_with_transforms = model
    elif weights_path.name.startswith("dense121_c5_"):
        model = torchvision.models.densenet121(
            num_classes=5,
        )
        model_with_transforms = ModelWithTransforms(
            model,
            torchvision.models.DenseNet121_Weights.IMAGENET1K_V1.transforms(
                antialias=True
            ),
        )
    else:
        raise RuntimeError("Unknown model")

    print("Loading weights from disk...")
    checkpoint = torch.load(PROJECT_ROOT / weights_path, weights_only=False)

    print("Restoring weights...")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    traced_module = torch.jit.script(model_with_transforms)
    traced_module.save(output_path)


def main() -> None:
    weights_paths = PROJECT_ROOT.glob("models/**/*.pth")
    for weights_path in weights_paths:
        output_path = PROJECT_ROOT / weights_path.with_suffix(".ptc")
        if not output_path.exists():
            compile_model(weights_path, output_path)


if __name__ == "__main__":
    main()
