# TLSA: LLM-Guided Text-Label Space Alignment with Contrastive Learning for Generalized Category Discovery

This is the implementation of our ACL 2026 paper TLSA. 

## Requirements

After creating a virtual environment, run

```bash
pip install -r requirements.txt
```

## Pretrained Model

Download the pretrained checkpoints and put them into a folder named `pretrained_models` in the root directory.

For English datasets, place the English BERT checkpoint under:

```text
pretrained_models/bert-base-uncased
```

For THUCNEWS, place the Chinese BERT checkpoint under:

```text
pretrained_models/bert-base-chinese
```

## LLM Configuration

Set the LLM configuration before running TLSA. Leave the values empty in released scripts and fill them with your own API information locally.

```bash
export LLM_MODEL=""
export LLM_BASE_URL=""
export LLM_API_KEY=""
```

## How to run

BANKING dataset as an example:

```bash
sh scripts/run_banking_0.5.sh
sh scripts/run_clinc_0.5.sh
sh scripts/run_hwu_0.5.sh
sh scripts/run_thucnews_0.5.sh
```

## How to cite

```bibtex
@inproceedings{xu-etal-2026-tlsa,
    title = "{TLSA}: {LLM}-Guided Text-Label Space Alignment with Contrastive Learning for Generalized Category Discovery",
    author = "Xu, Wenxi  and
      Qin, Chuan  and
      Chen, Xi  and
      Fang, Chuyu  and
      Zhou, Yuanchun  and
      Zhu, Hengshu",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.869/",
    pages = "19030--19046",
    ISBN = "979-8-89176-390-6",
    abstract = "Generalized Category Discovery (GCD) aims to classify data from partially labeled datasets by jointly recognizing known categories and discovering novel ones.Despite recent advances, existing methods still suffer from weak text{--}label alignment, inconsistent objectives across known and novel categories, and poor discrimination of semantically similar clusters. To mitigate these issues, we propose TLSA, a unified framework that enforces contrastive alignment between text and label representations within a shared semantic space. Specifically, we first design a label-semantic aware dual-encoder equipped with a symmetric contrastive objective to achieve text-label alignment. Then, we leverage LLM-based label induction to generate explicit and semantically meaningful names for previously unseen categories, followed by a graph-based refinement strategy that disambiguates semantically overlapping clusters through forced renaming. Finally, a confidence-aware sampling strategy ensures balanced learning across both easy and hard instances. Extensive experiments on four benchmark datasets show that TLSA consistently outperforms state-of-the-art GCD methods. The code is available at https://github.com/Wenxi-Xu/TLSA."
}
```
