# VLM2Vec (MMEB) for Pyserini

[![PyPI](https://img.shields.io/pypi/v/vlm2vec_for_pyserini?color=brightgreen)](https://pypi.org/project/vlm2vec_for_pyserini/)
[![Downloads](https://static.pepy.tech/personalized-badge/vlm2vec_for_pyserini?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/vlm2vec_for_pyserini)
[![Downloads](https://static.pepy.tech/personalized-badge/vlm2vec_for_pyserini?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/vlm2vec_for_pyserini)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


This repository contains a fork of the original VLM2Vec codebase, modified for easy [Pyserini](https://github.com/castorini/pyserini/) integration and repackaged as a PyPI package.

`current_version = "0.2.0"`

## Supported Datasets and Tasks
All 24 Visual Document Retrieval tasks are supported. This covers `ViDoRE`  `ViDoRE v2`, `VisRAG`, `ViDoSeek`, and `MMLongBench`.

## Supported Models
Any VL models with `qwen2-vl`, `gme`, and `lamra` backbones are supported. This includes `gme-Qwen2-VL-2B/7B-Instruct`, `VLM2Vec/VLM2Vec-V2.0`, `code-kunkun/LamRA-Ret` and more. 

## Installation

Install the package directly from PyPI:

```bash
pip install vlm2vec-for-pyserini
```

Or, install from source:

```bash
git clone https://github.com/castorini/VLM2Vec-for-Pyserini.git
cd VLM2Vec-for-Pyserini
pip install .
```

## Quick Start
Assuming that you have cloned the repository and you are in the root dir:
1. Download the visdoc from HuggingFace and convert the `corpus`, `topics` and `queries` to the format ready for Pyserini:
```bash
bash src/pyserini_integration/prepare_dataset.sh
```
2. Run encoding, indexing, and search. Then evaluation and results aggregation using the following script:
```bash
bash src/pyserini_integration/experiments.sh
```
If you want to use the PyPI package, take a look at `download_visdoc.py`, `save_pyserini_data.py`, and `quick_start_demo.py` files under `src/pyserini_integration/` as sample code.

## Contact
For contact regarding the Pyserini integration section, please email [Sahel Sharifymoghaddam](mailto:sahel.sharifymoghaddam@uwaterloo.ca).

For contact regarding the original VLM2Vec codebase, please email the authors of the original repository.

## Citation

If you use this work with Pyserini, please cite Pyserini in addition to the original VLM2Vec paper:

```bibtex
@INPROCEEDINGS{Lin_etal_SIGIR2021_Pyserini,
   author = "Jimmy Lin and Xueguang Ma and Sheng-Chieh Lin and Jheng-Hong Yang and Ronak Pradeep and Rodrigo Nogueira",
   title = "{Pyserini}: A {Python} Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations",
   booktitle = "Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)",
   year = 2021,
   pages = "2356--2362",
}

@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}

@article{meng2025vlm2vecv2,
  title={VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
  author={Rui Meng and Ziyan Jiang and Ye Liu and Mingyi Su and Xinyi Yang and Yuepeng Fu and Can Qin and Zeyuan Chen and Ran Xu and Caiming Xiong and Yingbo Zhou and Wenhu Chen and Semih Yavuz},
  journal={arXiv preprint arXiv:2507.04590},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
