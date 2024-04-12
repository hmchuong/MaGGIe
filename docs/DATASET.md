# DATASET

## Download

The dataset is available on [Hugging Face Datasets](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM). Please download the set you need:

| **Dataset**     | **Image/video** | **Train/val** | **Filenames**         |
|-----------------|-------------|-----------|-----------------------|
| I-HIM50K        | Image       | Train     | I-HIM50K.z* (6 files) |
| M-HIM2K + HIM2K | Image       | Val       | HIM2K_M-HIM2K.zip     |
| V-HIM2K5        | Video       | Train     | V-HIM2K5.z* (7 files) |
| V-HIM60         | Video       | Val       | V-HIM60.zip           |

For the zip splits, you can run following command to combine before unzip:
```
# e.g, if you want to combine I-HIM50K files
zip -F I-HIM50K.zip --out I-HIM50K_all.zip

# then unzip the file
unzip I-HIM50K_all.zip
```

## How to synthesize dataset?
Coming soon...