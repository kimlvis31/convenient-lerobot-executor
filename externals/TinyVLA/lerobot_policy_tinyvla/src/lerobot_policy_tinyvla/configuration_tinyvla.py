from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig


DEFAULT_WORKSPACE = "~/convLexecutor/externals/TinyVLA"


@PreTrainedConfig.register_subclass("tinyvla")
@dataclass
class TinyVLAConfig(PreTrainedConfig):
    device: str | None = "cpu"
    workspace: str = DEFAULT_WORKSPACE
    inference_python: str = ""
    step7_script: str = ""
    model_path: str = ""
    worker_log_dir: str = ""
    n_action_steps: int = 16
    seed: int = 0
    startup_timeout: float = 900.0
    inference_timeout: float = 180.0
    input_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    })
    output_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    })

    def __post_init__(self):
        super().__post_init__()
        if self.device != "cpu" or self.use_amp:
            raise ValueError("The LeRobot proxy must use device=cpu and use_amp=false.")
        if not 1 <= self.n_action_steps <= 16:
            raise ValueError("n_action_steps must be between 1 and 16.")
        if self.startup_timeout <= 0 or self.inference_timeout <= 0:
            raise ValueError("Timeouts must be positive.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(16))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("This adapter supports inference only.")

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        for key in ("observation.images.image", "observation.images.image2"):
            if key not in self.input_features or self.input_features[key].type != FeatureType.VISUAL:
                raise ValueError(f"Missing camera feature: {key}")
        if self.output_features["action"].shape != (7,):
            raise ValueError("TinyVLA requires seven action values.")
        if self.input_features["observation.state"].shape != (8,):
            raise ValueError("TinyVLA requires eight state values.")
