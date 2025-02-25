## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.
- For these who failed install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try `cd ctcdecode` and `pip install .`
- Pealse follow [this link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install pytorch geometric
- You can install other required modules by conducting
  `pip install -r requirements.txt`
  `pip install transformers`

## Data Preparation

1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

Download datasets and extract them, no further data preprocessing needed.

# SLR

### Weights

Here we provide the performance of the model and its corresponding weights.

| Dataset    | Backbone | Dev WER | Test WER | Pretrained model                                                                                                          |
| ---------- | -------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| Phoenix14  | Resnet34 | 17.00   | 17.76    | [[Google Drive]](https://drive.google.com/drive/folders/1GIRjSSunMGwgOp8JqlK3x7ERct4nv5tf?dmr=1&ec=wgc-drive-globalnav-goto) |
| Phoenix14T | Resnet34 | 15.96   | 18.48    | [[Google Drive]](https://drive.google.com/drive/folders/102_9th1pHyiv698qx6lznP7PMX07nIY3?dmr=1&ec=wgc-drive-globalnav-goto) |
| CSL-Daily  | Resnet34 | 25.14   | 23.89    | [[Google Drive]](https://drive.google.com/drive/folders/1xTAaS70KTHLajONwEJSL4npcWXguulwp?dmr=1&ec=wgc-drive-globalnav-goto) |

### Evaluate

To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py --load-weights path_to_weight.pt --phase test`

```
python main.py --load-weights ./phoenix2014-T/best_model.pt --phase test

python main.py --load-weights ./phoenix2014/best_model.pt --phase test

python main.py --load-weights ./csl-daily/best_model.pt --phase test
```

### Training

To Training the SignDAGC model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py `

Multi-machine training (In fact, the results of the Multi-machine run are not good):

`python -m torch.distributed.launch --nproc_per_node=2 main.py --device 0,1`

### Acknowledgments

Our code is based on [SignGraph](https://github.com/gswycf/SignGraph) and [GreedyViG](https://github.com/SLDGroup/GreedyViG).



# SLT

The code for the sign language translation part is visible [SignDAGC-SLT](https://github.com/xhonghu/SignDAGC_SLT-master).
