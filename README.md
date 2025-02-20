<div align="center">

# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding

[![Static Badge](https://img.shields.io/badge/arxiv-2501.18362-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2501.18362)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://medxpertqa.github.io)  [![Static Badge](https://img.shields.io/badge/licence-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/TsinghuaC3I/MedXpertQA/blob/main/LICENSE)

</div>


This is the official repository for the paper "MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding".

## 🔥 Updates

- **🎉 [2025-02-20] [Leaderboard](https://medxpertqa.github.io) is on! Check out the results of o3-mini, DeepSeek-R1, o1, and Qwen2.5-VL-72B!**
- **🤗 [2025-02-09] We release the MedXpertQA [dataset](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA).**
- **🔥 [2025-01-31] We introduce [MedXpertQA](https://arxiv.org/abs/2501.18362), a highly challenging and comprehensive benchmark to evaluate expert-level medical knowledge and advanced reasoning!**

## Overview

**MedXpertQA** includes 4,460 questions spanning 17 specialties and 11 body systems. It includes two subsets, **MedXpertQA Text** for text medical evaluation and **MedXpertQA MM** for multimodal medical evaluation. The following figure presents an overview. The left side illustrates MedXpertQA's diverse data sources, image types, and question attributes. The right side compares typical examples from MedXpertQA MM and a traditional multimodal medical benchmark (VQA-RAD).

<p align="center">
   <img src="figs/overview.png" alt="Overview of MedXpertQA." style="width: 90%;">
</p>


## Features

- **Next-Generation Multimodal Medical Evaluation:** MedXpert MM introduces expert-level medical exam questions with diverse images and rich clinical information, including patient records and examination results, setting it apart from traditional medical multimodal benchmarks with simple QA pairs generated from image captions.
- **Highly Challenging:** MedXpertQA introduces high-difficulty medical exam questions and applies rigorous filtering and augmentation, effectively addressing the insufficient difficulty of existing benchmarks like MedQA. The Text and MM subsets are currently the most challenging benchmarks in their respective fields.
- **Clinical Relevance:**  MedXpertQA incorporates specialty board questions to improve clinical relevance and comprehensiveness by collecting questions corresponding to 17/25 member board exams (specialties) of the American Board of Medical Specialties. It showcases remarkable diversity across multiple dimensions.

<p align="center">
   <img src="figs/diversity.png" alt="MedXpertQA spans diverse human body systems, medical tasks, and question topics." style="width: 90%;">
</p>

- **Mitigating Data Leakage:** We perform data synthesis to mitigate data leakage risk and conduct multiple rounds of expert reviews to ensure accuracy and reliability.
- **Reasoning-Oriented Evaluation:** Medicine provides a rich and representative setting for assessing reasoning abilities beyond mathematics and code. We develop a reasoning-oriented subset to facilitate the assessment of o1-like models.

## Leaderboard

We evaluate 16 leading proprietary and open-source LMMs and LLMs including advanced inference-time scaled models with a focus on the latest progress in medical reasoning capabilities.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="figs/leaderboard1.png" width="480px">
  <img src="figs/leaderboard2.png" width="265px">
</div>

## Contact

- Shang Qu: lindsay2864tt@gmail.com

- Ning Ding: dn97@mail.tsinghua.edu.cn

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/TsinghuaC3I/MedXpertQA/blob/main/LICENSE) file for details.

## Citation

If you find our work helpful, please use the following citation.

```bibtex
@article{zuo2025medxpertqa,
  title={MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding},
  author={Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2501.18362},
  year={2025}
}
```
