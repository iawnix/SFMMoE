# SFMMoE




# FILE Structure
```
├── data
│   ├── final_all_data.csv                                                          # Full dataset
│   └── Train-All.pt                                                                # Full dataset torch format
├── pt
│   └── kafnet-Parameter_v2-md-mmoe-cat-loss-All-seed1.pt                           # final model weight pt
├── model
│   ├── SF_Gatv2.py                                                                 # GATV2
│   ├── SF_GCN.py                                                                   # GCN
│   ├── SF_KAFMLP.py                                                                # KAFNet
│   └── SF_MMoE.py                                                                  # SFMMoE
├── util
│   ├── data_pyg.py                                                                 # datapoint
│   └── tools.py                                                                    # other tool for model
├── train
│   ├── explain
│   │   ├── 1_draw_explain.ipynb
│   │   ├── 2_draw_r2_embedding.ipynb
│   │   ├── 3_test_gap_indirect.ipynb
│   │   └── fig
│   │       ├── delat_MSE_gap1_2.png
│   │       ├── expert_task_gate_test.png
│   │       ├── expert_task_gate_train.png
│   │       ├── gap1.png
│   │       ├── gap2.png
│   │       ├── r2_test.png
│   │       ├── r2_test_s1-2t1.png
│   │       ├── r2_test_s1.png
│   │       ├── r2_test_t1.png
│   │       ├── r2_test_t2-2t1.png
│   │       ├── r2_test_t2.png
│   │       ├── s1.png
│   │       ├── t1.png
│   │       ├── t2.png
│   │       └── task_hop_train.png
│   ├── out
│   │   ├── log
│   │   │   └── Parameter-202504_19_03_10.log
│   │   └── Parameter-202504_19_03_10
│   │       ├── kafnet-Parameter_v2-md-mmoe-cat-loss-All-seed12.pt
│   │       ├── kafnet-Parameter_v2-md-mmoe-cat-loss-All-seed15.pt
│   │       ├── kafnet-Parameter_v2-md-mmoe-cat-loss-All-seed1.pt
│   │       ├── kafnet-Parameter_v2-md-mmoe-cat-loss-All-seed21.pt
│   │       └── README.md
│   ├── Parameter-202511_16_22_27
│   │   ├── SFMMoE-seed12.pt
│   │   ├── SFMMoE-seed15.pt
│   │   ├── SFMMoE-seed1.pt
│   │   └── SFMMoE-seed21.pt
│   ├── predict.py                                                                  # example for predict
│   └── train_SFMMoE.ipynb                                                          # example for train model
└── READMD.md



```
