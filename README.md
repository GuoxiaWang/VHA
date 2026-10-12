# Virtual Head Attention (VHA)

*Accepted to NeurIPS 2026; updated manuscript forthcoming.*

Virtual Head Attention (VHA) increases query-head diversity in compact grouped-query attention (GQA) during training, without requiring a new inference kernel. **Q Premix** expands physical query heads into virtual heads; **Linear Postmix** recombines their outputs. Both linear maps can be folded into standard GQA projections for inference. In the paper's two-KV-head setting, the resulting architecture is GQA-2.

This repository provides the VHA implementation in PaddleFleet and a small runnable training example. Set `use_vha_attention=True` in [`GPTConfig`](src/paddlefleet/models/gpt/gpt_config.py) to enable it for GPT layers.

The paper evaluates VHA's quality–efficiency trade-off across dense and MoE models from 0.6B to 30B-A3B, with a separate long-context study extending to 128K.

## Quick start

On Linux x86-64 with an NVIDIA GPU and a CUDA 13-compatible driver, run the VHA example with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --python 3.12
CUDA_VISIBLE_DEVICES=0 uv run --locked python examples/vha_tiny_gpt.py
```

The [example](examples/vha_tiny_gpt.py) trains a two-layer GPT in FP32 on synthetic tokens for two steps and checks finite losses, VHA gradients, and parameter updates. It is a functional smoke test, not a reproduction of the paper's results.

## Implementation

VHA is integrated at three points: [configuration](src/paddlefleet/transformer/transformer_config.py), [GPT layer selection](src/paddlefleet/models/gpt/gpt_layer_specs.py), and the [attention layer](src/paddlefleet/transformer/attention.py). The Python package keeps the `paddlefleet` import name.

## License

This code builds on PaddleFleet and is licensed under [Apache 2.0](LICENSE). Existing file-level copyright notices are preserved.
