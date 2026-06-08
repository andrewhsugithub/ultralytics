## SimAM Placement Configurations

> _See [rtdetr-l.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml) or [architecture](./assets/architecture) for detailed architecture._

> _See SimAM/HGBlock_SimAM implementation [here](./ultralytics/nn/AddModules/SimAM.py)._

| Case  | Placement                                      | Config                                                                                      | Training                                         | Logs                                               |
| ----- | ---------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| **1** | Baseline (B8) Unmodified RT-DETR-L, Batch 8.   | [rtdetr-l.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml)                             | [train-baseline.py](train-baseline.py)           | [Baseline batch 8](./logs/baseline-b8)             |
| **2** | Baseline (B16) Unmodified RT-DETR-L, Batch 16. | [rtdetr-l.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml)                             | [train-baseline.py](train-baseline.py)           | [Baseline batch 16](./logs/baseline-b16)           |
| **3** | Standalone SimAM before Stage 2 HGBlocks       | [rtdetr-l-SimAM-pre_s2.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l-SimAM-pre_s2.yaml)   | [train-simam-pre_s2.py](train-simam-pre_s2.py)   | [SimAM Pre S2 batch 8](./logs/simam-pre_s2-b8)     |
| **4** | Standalone SimAM after Stage 2 HGBlocks        | [rtdetr-l-SimAM-post_s2.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l-SimAM-post_s2.yaml) | [train-simam-post_s2.py](train-simam-post_s2.py) | [SimAM Post S2 batch 8](./logs/simam-post_s2-b8)   |
| **5** | Standalone SimAM before Stage 4 HGBlocks       | [rtdetr-l-SimAM.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l-SimAM.yaml)                 | [train-simam.py](train-simam.py)                 | [SimAM batch 8](./logs/simam-b8)                   |
| **6** | Stage 3 HGBlock_SimAM integration (B8)         | [rtdetr-l-HGBlock_SimAM.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l-HGBlock_SimAM.yaml) | [train-hgblock-simam.py](train-hgblock-simam.py) | [HGBlock SimAM batch 8](./logs/hgblock_simam-b8)   |
| **7** | Stage 3 HGBlock_SimAM integration (B16)        | [rtdetr-l-HGBlock_SimAM.yaml](./ultralytics/cfg/models/rt-detr/rtdetr-l-HGBlock_SimAM.yaml) | [train-hgblock-simam.py](train-hgblock-simam.py) | [HGBlock SimAM batch 16](./logs/hgblock_simam-b16) |

### Training Configuration

All runs utilized identical/default hyperparameters to ensure fair comparison:

- **Model:** RT-DETR-L (Pretrained COCO weights).
- **Dataset:** BCCD (255 train / 73 val / 36 test images). _Note: Due to the small dataset size (255 training images), results may be sensitive to data variance despite consistent training settings._
- **SimAM lambda:** 10^-4.
- **Epochs:** 50.
- **GPU**: NVIDIA RTX 4070
- All experiments used identical training hyperparameters (learning rate, optimizer, augmentation pipeline) to ensure a controlled comparison across configurations.

> _See [bccd.yaml](./ultralytics/cfg/datasets/bccd.yaml) for detailed configuration._

---

## Experimental Results

_All results can be seen at [assets](./assets/)._

_Note: **Bold** values indicate the best metric across all
configurations._

### Overall Detection Performance Across SimAM Placement Configurations

Overall Detection Performance Across SimAM Placement Configurations:

| Case | Configuration     | Batch | Precision    | Recall       | mAP50      | mAP50-95     | Params |
| ---- | ----------------- | ----- | ------------ | ------------ | ---------- | ------------ | ------ |
| 1    | Baseline          | 8     | 0.808        | <u>0.896</u> | 0.891      | 0.638        | ~31.9M |
| 2    | Baseline          | 16    | **0.835**    | 0.87         | 0.884      | 0.639        | ~31.9M |
| 3    | SimAM (Before S2) | 8     | 0.705        | 0.844        | 0.83       | 0.584        | ~31.9M |
| 4    | SimAM (After S2)  | 8     | 0.723        | 0.816        | 0.818      | 0.58         | ~31.9M |
| 5    | SimAM (Before S4) | 8     | 0.804        | 0.832        | 0.886      | 0.633        | ~31.9M |
| 6    | HGBlock SimAM     | 8     | 0.795        | **0.898**    | <u>0.9</u> | <u>0.643</u> | ~31.9M |
| 7    | HGBlock SimAM     | 16    | <u>0.825</u> | 0.883        | **0.905**  | **0.648**    | ~31.9M |

### Class-Specific Analysis: Small Object Enhancement (Platelets)

Red Blood Cell (RBC) Detection Metrics by Configuration:

| Case | Configuration     | Batch | Precision    | Recall       | mAP50        | mAP50-95     | Params |
| ---- | ----------------- | ----- | ------------ | ------------ | ------------ | ------------ | ------ |
| 1    | Baseline          | 8     | 0.663        | <u>0.832</u> | 0.826        | 0.601        | ~31.9M |
| 2    | Baseline          | 16    | <u>0.708</u> | 0.779        | 0.826        | <u>0.605</u> | ~31.9M |
| 3    | SimAM (Before S2) | 8     | 0.584        | 0.762        | 0.72         | 0.512        | ~31.9M |
| 4    | SimAM (After S2)  | 8     | 0.637        | 0.746        | 0.77         | 0.557        | ~31.9M |
| 5    | SimAM (Before S4) | 8     | **0.738**    | 0.769        | **0.836**    | 0.603        | ~31.9M |
| 6    | HGBlock SimAM     | 8     | 0.624        | **0.836**    | 0.82         | 0.597        | ~31.9M |
| 7    | HGBlock SimAM     | 16    | 0.681        | 0.819        | <u>0.833</u> | **0.607**    | ~31.9M |

White Blood Cell (WBC) Detection Metrics by Configuration:

| Case | Configuration     | Batch | Precision    | Recall       | mAP50        | mAP50-95     | Params |
| ---- | ----------------- | ----- | ------------ | ------------ | ------------ | ------------ | ------ |
| 1    | Baseline          | 8     | **0.97**     | **1**        | 0.975        | **0.876**    | ~31.9M |
| 2    | Baseline          | 16    | <u>0.968</u> | **1**        | 0.977        | <u>0.815</u> | ~31.9M |
| 3    | SimAM (Before S2) | 8     | 0.803        | **1**        | 0.965        | 0.772        | ~31.9M |
| 4    | SimAM (After S2)  | 8     | 0.846        | <u>0.993</u> | 0.957        | 0.772        | ~31.9M |
| 5    | SimAM (Before S4) | 8     | 0.861        | 0.986        | 0.968        | 0.797        | ~31.9M |
| 6    | HGBlock SimAM     | 8     | 0.964        | 0.986        | **0.982**    | 0.814        | ~31.9M |
| 7    | HGBlock SimAM     | 16    | 0.966        | **1**        | <u>0.978</u> | 0.808        | ~31.9M |

Platelets Detection Metrics by Configuration:

| Case | Configuration     | Batch | Precision    | Recall       | mAP50        | mAP50-95     | Params |
| ---- | ----------------- | ----- | ------------ | ------------ | ------------ | ------------ | ------ |
| 1    | Baseline          | 8     | 0.79         | <u>0.855</u> | 0.872        | 0.497        | ~31.9M |
| 2    | Baseline          | 16    | **0.829**    | 0.83         | 0.848        | 0.496        | ~31.9M |
| 3    | SimAM (Before S2) | 8     | 0.727        | 0.769        | 0.805        | 0.468        | ~31.9M |
| 4    | SimAM (After S2)  | 8     | 0.687        | 0.711        | 0.727        | 0.411        | ~31.9M |
| 5    | SimAM (Before S4) | 8     | <u>0.812</u> | 0.739        | 0.855        | 0.498        | ~31.9M |
| 6    | HGBlock SimAM     | 8     | 0.796        | **0.873**    | <u>0.899</u> | <u>0.519</u> | ~31.9M |
| 7    | HGBlock SimAM     | 16    | **0.829**    | 0.829        | **0.903**    | **0.528**    | ~31.9M |

Class-Specific mAP50 Comparison Highlighting Small Object Enhancement (Platelets):
| Class | Case 1 | Case 7 | Δ mAP50 |
|:------------|:-------------:|:---------------------:|:--------|
| **Platelets** (Small, 8–15px) | 0.872 | **0.903** | **+0.031** |
| RBCs (Medium) | 0.826 | <u>0.833</u> | +0.007 |
| WBCs (Large) | 0.975 | <u>0.978</u> | +0.003 |
