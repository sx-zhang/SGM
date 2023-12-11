# Imagine Before Go: Self-Supervised Generative Map for Object Goal Navigation
<!-- Sixian Zhang, Xinyao Yu, Xinhang Song, Xiaohan Wang, Shuqiang Jiang  -->

## Setup
- Clone the repository and move into the top-level directory `cd SGM`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate sgm`
- We provide pre-trained model of [sgm](....) and [area_prediction](....). For evaluation, you can download them to the directory.
- Download the [sgm_dataset](....).  

## Dataset
We use a modified version of the Gibson ObjectNav evaluation setup from [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation).

1. Download the [Gibson ObjectNav dataset](https://utexas.box.com/s/tss7udt3ralioalb6eskj3z3spuvwz7v) to `$SGM_ROOT/data/datasets/objectnav/gibson`.
    ```
    cd $SGM_ROOT/data/datasets/objectnav
    wget -O gibson_objectnav_episodes.tar.gz https://utexas.box.com/shared/static/tss7udt3ralioalb6eskj3z3spuvwz7v.gz
    tar -xvzf gibson_objectnav_episodes.tar.gz && rm gibson_objectnav_episodes.tar.gz
    ```
2. Download the image segmentation model [[URL](https://utexas.box.com/s/sf4prmup4fsiu6taljnt5ht8unev5ikq)] to `$SGM_ROOT/pretrained_models`.
3. To visualize episodes with the semantic map and potential function predictions, add the arguments `--print_images 1 --num_pf_maps 3` in the evaluation script.

The `data` folder should look like this
```python
  data/ 
    ├── datasets/objectnav/gibdon/v1.1
        ├── train/
        │   ├── content/
        │   ├── train_info.pbz2
        │   └── train.json.gz
        ├── val/
        │   ├── content/
        │   ├── val_info.pbz2
        │   └── val.json.gz
    ├── scene_datasets/
        ├── gibson_semantic/
            ├── Allensville_semantic.ply
            ├── Allensville.glb
            ├── Allensville.ids
            ├── Allensville.navmesh
            ├── Allensville.scn
            ├── ...
    ├── semantic_maps/
        ├── gibson/semantic_maps
            ├── semmap_GT_info.json
            ├── Allensville_0.png
            ├── Allensville.h5
            ├── ...
```

<!-- ## Training and Evaluation -->
<!-- ### Train our SGM model  -->

## Evaluation 
`sh experiment_scripts/gibson/eval_sgm.sh`

<!-- ## Citing
If you find this project useful in your research, please consider citing: -->
