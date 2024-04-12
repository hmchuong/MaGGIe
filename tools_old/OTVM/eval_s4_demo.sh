#!/bin/bash

GPU=$1

python demo_cvpr24.py --gpu $GPU --demo --data-dir /home/chuongh/vm2m/data/syn/benchmark/$2 --out-dir output/xmem_finetuned/$2