# SiFT:A Serial Framework with Textual Guidance for Federated Learning

This is the implementation of MICCAI 2024 paper ***SiFT:A Serial Framework with Textual Guidance for Federated Learning***

## Links
- [paper](https://link.springer.com/content/pdf/10.1007/978-3-031-72117-5_61)
- [github version]()

## Run the code
### Prepare the environment(recommended environment)
```python
torch==1.8.2
torchvision==0.9.2
numpy==1.21.6
medmnist==3.0.1
timm==0.6.13
scikit-learn==1.0.2
pandas==1.3.5
```

### Dataset
It is recommended to follow the links, download the files locally, and replace the dataset path argument in the option files.

- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [OrganCMNIST](https://zenodo.org/records/10519652/files/organcmnist_224.npz?download=1)
- [OrganSMNIST](https://zenodo.org/records/10519652/files/organsmnist_224.npz?download=1)


> **NOTE**:
> The OrganCMNIST and OrganSMNIST are integral components of the [MedMNIST](https://medmnist.com/).
Thanks for their hard work.

### Pre-trained Biomedical Language Model

We have provided a set of weight vectors generated by BioLinkBERT-large. If you wish to create your own weight vectors, you should download the weight file by this [link](https://huggingface.co/michiyasunaga/BioLinkBERT-large) and set the variable *bert_path* in `utils/text_translator.py` accordingly.

### Run the main.py
Create a terminal and run the main.py, ensuring that you have specified the config file.
Just like this:
```bash
python main.py -cff config-file-path
``` 

## Cite this paper
```text
@inproceedings{li2024sift,
  title={SiFT: A Serial Framework with Textual Guidance for Federated Learning},
  author={Li, Xuyang and Zhang, Weizhuo and Yu, Yue and Zheng, Wei-Shi and Zhang, Tong and Wang, Ruixuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={655--665},
  year={2024},
  organization={Springer}
}
```