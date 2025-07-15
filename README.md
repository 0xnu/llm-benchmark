## llm-benchmark

[![Lint](https://github.com/0xnu/llm-benchmark/actions/workflows/lint.yaml/badge.svg)](https://github.com/0xnu/llm-benchmark/actions/workflows/lint.yaml)
[![Release](https://img.shields.io/github/release/0xnu/llm-benchmark.svg)](https://github.com/0xnu/llm-benchmark/releases/latest)
[![License](https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53)](/LICENSE)

Test and compare different large language models on various tasks.

### Tasks

+ Code Generation
+ Mathematical Reasoning
+ Creative Writing
+ Data Analysis
+ Logical Reasoning
+ Summarisation
+ Technical Explanation
+ Problem Solving

### Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following

```python
## Prerequisites
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python3 -m pip install --upgrade pip
```

### LLM Benchmark

```python
## Set Environment Variables
OPENAI_API_KEY="your_api_key"
ANTHROPIC_API_KEY="your_api_key"
XAI_API_KEY="your_api_key"
DEEPSEEK_TOKEN="your_api_key"
GOOGLE_API_KEY="your_api_key"
MOONSHOT_API_KEY="your_api_key"
OPENROUTER_API_KEY="your_api_key"
MISTRAL_API_KEY="your_api_key"

## View Setup Guide
python3 -m scripts.benchmark setup

## Execute The Benchmark
python3 -m scripts.benchmark

## Deactivate Virtual Environment
deactivate
```

### Results

Generated results on the 15th July 2025.

```sh
================================================================================
LLM BENCHMARK SUMMARY
================================================================================
                                    Avg Quality  Quality Std  Avg Latency  Med Latency  Avg Cost  Total Cost  Error Rate
model_name                                                                                                              
Anthropic-claude-sonnet-4-20250514       89.376       12.846       15.611       11.989     0.018       0.294        0.00
DeepSeek-deepseek-chat                   89.476       10.643       47.782       43.808     0.001       0.014        0.00
Google-gemini-2.5-pro                    79.990       25.111       25.630       22.686     0.002       0.024        6.25
Mistral-mistral-medium-2505              87.411       13.699       15.303       12.847     0.005       0.076        0.00
Moonshot-moonshot-v1-8k                  89.498       13.530        7.811        7.455     0.001       0.014        0.00
OpenAI-gpt-4.1                           89.388       12.633        9.624        7.680     0.009       0.149        0.00
Qwen-qwen3-32b                           43.802       46.133       50.752       48.419     0.000       0.000        0.00
xAI-grok-4-0709                          91.358        9.587       27.256       27.061     0.025       0.395        0.00
================================================================================

Best Overall Quality: xAI-grok-4-0709
Fastest Response: Moonshot-moonshot-v1-8k
Most Cost-Effective: Qwen-qwen3-32b

📁 All results saved to results directory
🏆 Best Overall Model: xAI-grok-4-0709
📈 Overall Average Quality: 82.5%
💰 Total Cost: $0.9652
⚡ Average Latency: 24.97s
```

### License

This project is licensed under the [Modified MIT License](./LICENSE).

### Citation

```tex
@misc{llmbenchmark,
  author       = {Oketunji, A.F.},
  title        = {LLM Benchmark},
  year         = 2025,
  version      = {0.0.6},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15937532},
  url          = {https://doi.org/10.5281/zenodo.15937532}
}
```

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
