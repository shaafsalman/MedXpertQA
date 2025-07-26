<div align="center">

# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding

[![Static Badge](https://img.shields.io/badge/arxiv-2501.18362-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2501.18362)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://medxpertqa.github.io)  [![Static Badge](https://img.shields.io/badge/license-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/TsinghuaC3I/MedXpertQA/blob/main/LICENSE)

</div>

<div align="center">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🔥 News</a> •
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📖 Overview</a> •
    <a href="#features" style="text-decoration: none; font-weight: bold;">✨ Features</a> •
    <a href="#leaderboard" style="text-decoration: none; font-weight: bold;">📊 Leaderboard</a>
  </p>
  <p>
    <a href="#usage" style="text-decoration: none; font-weight: bold;">🔧 Usage</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

## 🔥News

- **🎉 [2025-05-06] MedXpertQA paper is accepted to [ICML 2025](https://icml.cc/Conferences/2025)!**
- **🛠️ [2025-04-08] MedXpertQA has been successfully integrated into [OpenCompass](https://github.com/open-compass/opencompass)! Check out the [PR](https://github.com/open-compass/opencompass/pull/2002)!**
- **💻 [2025-02-28] We release the evaluation code! Check out the [Usage](#usage).**
- **🌟 [2025-02-20] [Leaderboard](https://medxpertqa.github.io) is on! Check out the results of o3-mini, DeepSeek-R1, and o1!**
- **🤗 [2025-02-09] We release the MedXpertQA [dataset](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA).**
- **🔥 [2025-01-31] We introduce [MedXpertQA](https://arxiv.org/abs/2501.18362), a highly challenging and comprehensive benchmark to evaluate expert-level medical knowledge and advanced reasoning!**

## 📖Overview

**MedXpertQA** includes 4,460 questions spanning 17 specialties and 11 body systems. It includes two subsets, **MedXpertQA Text** for text medical evaluation and **MedXpertQA MM** for multimodal medical evaluation. The following figure presents an overview. 
<details>
<summary>
  More Details
</summary>
The left side illustrates the diverse data sources, image types, and question attributes.
The right side compares typical examples from MedXpertQA MM and a traditional benchmark (VQA-RAD).
</details>

<p align="center">
   <img src="figs/overview.png" alt="Overview of MedXpertQA." width="90%">
</p>


## ✨Features

- **Next-Generation Multimodal Medical Evaluation:** MedXpertQA MM introduces expert-level medical exam questions with diverse images and rich clinical information, including patient records and examination results, setting it apart from traditional medical multimodal benchmarks with simple QA pairs generated from image captions.
- **Highly Challenging:** MedXpertQA introduces high-difficulty medical exam questions and applies rigorous filtering and augmentation, effectively addressing the insufficient difficulty of existing benchmarks like MedQA. The Text and MM subsets are currently the most challenging benchmarks in their respective fields.
- **Clinical Relevance:**  MedXpertQA incorporates specialty board questions to improve clinical relevance and comprehensiveness by collecting questions corresponding to 17/25 member board exams (specialties) of the American Board of Medical Specialties. It showcases remarkable diversity across multiple dimensions.

<p align="center">
   <img src="figs/diversity.png" alt="MedXpertQA spans diverse human body systems, medical tasks, and question topics." width="90%">
</p>

- **Mitigating Data Leakage:** We perform data synthesis to mitigate data leakage risk and conduct multiple rounds of expert reviews to ensure accuracy and reliability.
- **Reasoning-Oriented Evaluation:** Medicine provides a rich and representative setting for assessing reasoning abilities beyond mathematics and code. We develop a reasoning-oriented subset to facilitate the assessment of o1-like models.

## 📊Leaderboard

We evaluate 17 leading proprietary and open-source LMMs and LLMs including advanced inference-time scaled models with a focus on the latest progress in medical reasoning capabilities.
**Further details are available in the [leaderboard](https://medxpertqa.github.io) and the [paper](https://arxiv.org/abs/2501.18362).**

<p align="center">
  <img src="figs/leaderboard1.png" width="60%">
  <img src="figs/leaderboard2.png" width="32.5%">
</p>


## 🔧Usage

1. Clone the Repository:

```
git clone https://github.com/TsinghuaC3I/MedXpertQA
cd MedXpertQA/eval
```

2. Install Dependencies:

```
pip3 install -r requirements.txt
```

3. Inference:

```
bash scripts/run.sh
```

> The *run.sh* script performs inference by calling *main.py*, which offers additional features such as multithreading. Additionally, you can modify *model/api_agent.py* to support more models.


## Set up for Custom Models

### Adding Your Own Model

To evaluate your custom model on MedXpertQA, follow these steps:

#### Step 1: Setup Dataset
```bash
cd eval/data
python setup_datasets.py  
```

#### Step 2: Add Model to Configuration
Edit `config/model_info.json` and add your model name to the `API_MODEL` list:

```json
{
    "API_MODEL": [
        "existing-models...",
        "your-custom-model-name",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "your-organization/your-model-name"
    ]
}
```

#### Step 3: Implement Model Interface
Add your model implementation in `model/api_agent.py`. Add a new condition in the `__init__` method:

```python
elif model_name in [
    "your-custom-model-name",
    "your-organization/your-model-name"
]:
    print("Custom Model")
    # For vLLM-based models:
    from vllm import LLM, SamplingParams
    import os
    
    os.environ['VLLM_USE_V1'] = '0'
    
    self.client = LLM(
        model=model_name,
        tensor_parallel_size=4,
        worker_use_ray=False,
        gpu_memory_utilization=0.75,
        trust_remote_code=True
    )
    self.sampling_params = SamplingParams(
        max_tokens=self.max_tokens, 
        temperature=self.temperature if self.temperature > 0 else 0.15
    )
    self.is_vllm = True
    
    # For API-based models:
    # self.client = OpenAI(
    #     api_key="your-api-key",
    #     base_url="your-api-endpoint",
    # )
```

#### Step 4: Create Custom Evaluation Script
Create `scripts/run_custom.sh` for your model:

```bash
#!/bin/bash
set -e
models=${1:-"your-custom-model-name"}
datasets=${2:-"medxpertqa"}
tasks=${3:-"text,mm"}
output_dir=${4:-"dev"}
method=${5:-"zero_shot"}
prompting_type=${6:-"cot"}
temperature=${7:-0}

# Sequential execution to avoid memory issues
IFS=","
for model in $models; do
    for dataset in $datasets; do
        for task in $tasks; do
            echo "Running: Model=${model}, Dataset=${dataset}, Task=${task}"
            
            python main.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --task "${task}" \
                --output-dir "${output_dir}" \
                --method "${method}" \
                --prompting-type "${prompting_type}" \
                --temperature "${temperature}" \
                --num-threads 1
            
            # Clean GPU memory between tasks
            python -c "import torch; torch.cuda.empty_cache()"
            sleep 2
        done
    done
done
```

#### Step 5: Run Evaluation
```bash
# Make script executable
chmod +x scripts/run_custom.sh

# Run evaluation
bash scripts/run_custom.sh "your-custom-model-name"

```



4. Evaluation:

We provide a script *eval.ipynb* to calculate accuracy on each subset.

> [!NOTE]
> Please use this script when evaluating the **QVQ** and **DeepSeek-R1**. Through case studies, we found that the answer cleaning function in the *utils.py* is unsuitable for these two models.



## 📨Contact

- Shang Qu: [lindsay2864tt@gmail.com](mailto:lindsay2864tt@gmail.com)

- Ning Ding: [dn97@mail.tsinghua.edu.cn](mailto:dn97@mail.tsinghua.edu.cn)

## ⚖️License

This project is licensed under the [MIT License](https://github.com/TsinghuaC3I/MedXpertQA/blob/main/LICENSE).

## 🎈Citation

If you find our work helpful, please use the following citation.

```bibtex
@article{zuo2025medxpertqa,
  title={Medxpertqa: Benchmarking expert-level medical reasoning and understanding},
  author={Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2501.18362},
  year={2025}
}
```
