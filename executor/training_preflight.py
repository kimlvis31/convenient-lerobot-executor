"""Check local native training inputs before starting LeRobot."""

import json
from pathlib import Path


def _read_optional_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Could not read training metadata ({path}).") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in training metadata ({path}).")
    return value


def _dataset_cameras(features):
    cameras = {}
    for name, feature in features.items():
        if not isinstance(feature, dict) or feature.get("dtype") not in {"image", "video"}:
            continue
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"Camera input {name} must have three dimensions.")
        dimension_names = feature.get("names")
        if isinstance(dimension_names, list) and dimension_names[-1:] in (["channel"], ["channels"]):
            shape = [shape[2], shape[0], shape[1]]
        cameras[name] = tuple(shape)
    return cameras


def _policy_cameras(features):
    return {
        name: tuple(feature.get("shape") or ())
        for name, feature in features.items()
        if isinstance(feature, dict) and feature.get("type") == "VISUAL"
    }


def _valid_rename_map(mapping):
    return isinstance(mapping, dict) and all(
        isinstance(source, str) and source and isinstance(target, str) and target
        for source, target in mapping.items()
    )


def _manual_rename_map(extra):
    tokens = [token for token in extra if isinstance(token, str) and token.startswith("--rename_map=")]
    if not tokens:
        return None
    try:
        mapping = json.loads(tokens[-1].split("=", 1)[1])
    except ValueError as error:
        raise ValueError("Use a JSON object with --rename_map.") from error
    if not _valid_rename_map(mapping):
        raise ValueError("--rename_map must map feature names to feature names.")
    return mapping


def _unique_names(features, mapping):
    return len({mapping.get(name, name) for name in features}) == len(features)


def _saved_camera_map(checkpoint, dataset, features, cameras, expected):
    saved = _read_optional_object(checkpoint / "train_config.json")
    saved_dataset = saved.get("dataset")
    mapping = saved.get("rename_map")
    if not isinstance(saved_dataset, dict) or not dataset or saved_dataset.get("repo_id") != dataset:
        return {}
    if not _valid_rename_map(mapping):
        return {}

    camera_map = {source: target for source, target in mapping.items() if source in cameras and target in expected}
    if not camera_map or not _unique_names(features, camera_map):
        return {}
    if any(expected[target] and cameras[source][0] != expected[target][0] for source, target in camera_map.items()):
        return {}
    if not expected.keys() <= {camera_map.get(name, name) for name in cameras}:
        return {}
    return camera_map


def _check_camera_constraints(model, cameras):
    if model == "vqbet" and len(cameras) != 1:
        raise ValueError(f"VQBeT requires exactly one camera input. Found {len(cameras)}.")
    if model == "tdmpc":
        if len(cameras) > 1:
            raise ValueError("TD-MPC supports at most one camera input.")
        if any(len(shape) == 3 and shape[-2] != shape[-1] for shape in cameras.values()):
            raise ValueError("TD-MPC requires square camera images.")
    if model in {"diffusion", "multi_task_dit"} and len({shape for shape in cameras.values() if shape}) > 1:
        raise ValueError(f"{model} requires camera inputs with matching image shapes.")


def native_training_issue(model):
    if model == "gaussian_actor":
        return ("Gaussian Actor uses LeRobot's separate reinforcement-learning actor/learner launcher. "
                "The supervised lerobot-train command does not implement its training objective. "
                "Use Evaluation here, or train with the reinforcement-learning launcher first.")
    return None


def wall_oss_camera_defaults(train, cameras, policy=None, manual=None):
    """Use the documented front/hand cameras of the standard LIBERO dataset."""
    explicit = any(isinstance(token, str) and token.startswith(
        ("--policy.camera_key_mapping=", "--policy.camera_key_mapping.")) for token in train.get("extra", ()))
    existing = (policy or {}).get("camera_key_mapping")
    if isinstance(existing, dict) and existing and set(existing) <= set(cameras):
        return {}
    known = {"observation.images.image": "face_view", "observation.images.image2": "right_wrist_view"}
    if (train.get("dataset") == "lerobot/libero" and set(cameras) == set(known)
            and manual is None and not explicit):
        return {"policy.camera_key_mapping": known}
    return {}


def molmo_feature_defaults(train, dataset, policy=None):
    """Name only the documented coarse-labelled standard LIBERO feature layout."""
    policy = policy or {}
    extra = train.get("extra", ())
    if (train.get("dataset") not in {"lerobot/libero", "HuggingFaceVLA/libero"}
            or dataset.get("robot_type") != "panda"
            or any(isinstance(token, str) and token.startswith(
                ("--policy.dataset_feature_names=", "--policy.dataset_feature_names.")) for token in extra)
            or policy.get("normalize_gripper") is True
            or "--policy.normalize_gripper=true" in extra):
        return {}
    features = dataset.get("features", {})
    documented = {
        "action": ["delta_x", "delta_y", "delta_z", "delta_axis_angle_x", "delta_axis_angle_y",
                   "delta_axis_angle_z", "gripper"],
        "observation.state": ["eef_x", "eef_y", "eef_z", "eef_axis_angle_x", "eef_axis_angle_y",
                              "eef_axis_angle_z", "gripper_left", "gripper_right"],
    }
    if any(features.get(key, {}).get("shape") != [len(names)]
           or features.get(key, {}).get("dtype") != "float32" for key, names in documented.items()):
        return {}
    # These labels describe the target dataset. Native Molmo overwrites saved
    # source-dataset labels during construction, so a coarse target needs its
    # own documented dimension names even when the source labels were complete.
    names = {}
    coarse = {"action": (["actions"], None), "observation.state": (["state"], ["states"], None)}
    for key, known in documented.items():
        if features[key].get("names") in coarse[key]:
            names[key] = known
        elif isinstance(features[key].get("names"), list) and len(features[key]["names"]) == len(known):
            names[key] = features[key]["names"]
    coarse_target = any(features[key].get("names") in coarse[key] for key in documented)
    return {"policy.dataset_feature_names": names} if coarse_target and len(names) == len(documented) else {}


def molmo_camera_map(train, dataset, cameras, expected):
    source = {"observation.images.image", "observation.images.image2"}
    target = {"observation.images.image", "observation.images.wrist_image"}
    if (train.get("dataset") in {"lerobot/libero", "HuggingFaceVLA/libero"}
            and dataset.get("robot_type") == "panda" and set(cameras) == source and set(expected) == target
            and all(cameras[name][0] == expected["observation.images.image" if name.endswith(".image")
                                                    else "observation.images.wrist_image"][0] for name in source)):
        return {"observation.images.image2": "observation.images.wrist_image"}
    return {}


def native_training_preflight(config, *, policy_config=None):
    """Return required automatic flags for a native job using known local metadata."""
    train = config["train"]
    if issue := native_training_issue(config["model"]):
        raise ValueError(issue)
    if train.get("resume"):
        return {}

    model = config["model"]
    if model == "vqbet" and train.get("device") == "mps":
        raise ValueError("VQBeT does not support the mps device. Select cpu or cuda.")

    extra = train.get("extra", [])
    manual = _manual_rename_map(extra)
    root = train.get("dataset_root")
    if not root:
        return {}
    dataset = _read_optional_object(Path(root).expanduser() / "meta/info.json")
    features = dataset.get("features")
    if not isinstance(features, dict) or not features:
        return {}
    if manual is not None and not _unique_names(features, manual):
        raise ValueError("--rename_map maps multiple dataset features to the same name.")
    if any(isinstance(token, str) and token.startswith(("--rename_map.", "--policy.input_features=",
                                                        "--policy.input_features.")) for token in extra):
        return {}

    auto = molmo_feature_defaults(train, dataset, policy_config) if model == "molmoact2" else {}
    cameras = _dataset_cameras(features)
    pretrained = train.get("pretrained")
    if pretrained:
        checkpoint = Path(pretrained).expanduser()
        policy = policy_config if policy_config is not None else _read_optional_object(checkpoint / "config.json")
        if not policy:
            return {}
        if policy.get("type") and policy["type"] != model:
            raise ValueError(f"The selected checkpoint uses policy {policy['type']}, but this job uses {model}.")
        if model == "molmoact2":
            auto = molmo_feature_defaults(train, dataset, policy)
        if model == "wall_oss_05":
            # Wall-OSS derives its input features from the target dataset and resolves
            # physical camera slots using its own camera_key_mapping overrides.
            return wall_oss_camera_defaults(train, cameras, policy, manual)
        inputs = policy.get("input_features")
        if inputs and isinstance(inputs, dict):
            expected = _policy_cameras(inputs)
            provided = {manual.get(name, name) for name in cameras} if manual else cameras.keys()
            # LeRobot 145cc7ea accepts either visual feature set as a subset of the other.
            if not (expected.keys() <= provided or expected.keys() >= provided):
                if manual is None:
                    mapping = _saved_camera_map(checkpoint, train.get("dataset"), features, cameras, expected)
                    if not mapping and model == "molmoact2":
                        mapping = molmo_camera_map(train, dataset, cameras, expected)
                    if mapping:
                        auto["rename_map"] = mapping
                if "rename_map" not in auto:
                    missing = ", ".join(sorted(expected.keys() - provided))
                    unexpected = ", ".join(sorted(provided - expected.keys()))
                    raise ValueError(f"{model} checkpoint camera inputs do not match this dataset. "
                                     f"Missing {missing}. Unexpected {unexpected}.")
            cameras = expected
    if model == "wall_oss_05":
        return wall_oss_camera_defaults(train, cameras, manual=manual)
    _check_camera_constraints(model, cameras)
    return auto
