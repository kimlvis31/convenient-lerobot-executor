# Third-party sources

| Component | Source and attribution |
| --- | --- |
| LeRobot | [Hugging Face LeRobot](https://github.com/huggingface/lerobot), revision [`145cc7e`](https://github.com/huggingface/lerobot/commit/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8). Installed separately. |
| TinyVLA source bundle | [Midea AI TinyVLA](https://github.com/midea-ai/TinyVLA). The bundled subset lives under `externals/TinyVLA/sources/TinyVLA/` with its original `LICENSE` and the nested `policy_heads/LICENSE`. It omits standalone training and evaluation scripts and machine-specific encoder test blocks. The action head allocates inference samples using the actual batch dimension. |
| Llava-Pythia model | [lesjie/Llava-Pythia-700M](https://huggingface.co/lesjie/Llava-Pythia-700M), revision `9ac93b5370af5a0b6e66ffa2b60e0316489826d5`. The model card specifies MIT. Configuration files are included. The weights and tokenizer vocabulary are fetched separately and checked against the manifest. |
| LIBERO example | [lerobot/libero](https://huggingface.co/datasets/lerobot/libero), revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The dataset card lists Apache-2.0. This project records the selected episode and file manifest and fetches data at setup time. |
