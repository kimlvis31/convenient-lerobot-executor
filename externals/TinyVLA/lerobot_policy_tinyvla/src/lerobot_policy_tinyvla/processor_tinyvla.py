from lerobot.processor import make_policy_processor_pipelines


def make_tinyvla_pre_post_processors(config, dataset_stats=None, dataset_meta=None):
    # LIBERO's environment processor supplies the state vector and camera rotation.
    # The TinyVLA process owns its image processor; state/action use identity scaling.
    return make_policy_processor_pipelines(input_steps=[], output_steps=[])
