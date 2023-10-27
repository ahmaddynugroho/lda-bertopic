### About

Comparing Gensim's LDA and BERTopic for topic modeling for dataset with different length and preprocessing

### Reproduce

```
mamba env create -f environment.yml
mamba activate ldab

# prepare dataset and result directory
mkdir datasets datasets/ds results

# get dataset
curl -Lo datasets/articles.csv https://github.com/ahmaddynugroho/lda-bertopic/releases/download/ds_and_result/articles.csv

# Create dataset variations (raw and preprocessed)
# You may also configure how much documents to processed in this script
# or you can skip this by downloading the preprocessed datasets:
# curl -Lo datasets/ds.tar.bz https://github.com/ahmaddynugroho/lda-bertopic/releases/download/ds_and_result/ds.tar.bz
# tar xvjf datasets/ds.tar.bz -C datasets
python ds.py

# Main script to compare both algorithms with datasets variations
# This script only uses CPU since Gensim only supports it
python cpu.py
```

### Tested on
```yml
OS: Ubuntu 22.04.2 LTS on Windows 10 x86_64
Kernel: 5.15.90.1-microsoft-standard-WSL2
Shell: bash 5.1.16
Terminal: Windows Terminal
CPU: AMD Ryzen 3 2200G (4) @ 3.493GHz
GPU: Radeon™ Vega 8 Graphics
```
