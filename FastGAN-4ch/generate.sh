#!/bin/bash

' Commented
python generate_4ch.py --dist '/work/vajira/DATA/FastGAN_polyps/generated_1k_set_0' --save_option "image_only"
python generate_4ch.py --dist '/work/vajira/DATA/FastGAN_polyps/generated_1k_set_1' --save_option "image_only"
python generate_4ch.py --dist '/work/vajira/DATA/FastGAN_polyps/generated_1k_set_2' --save_option "image_only"
python generate_4ch.py --dist '/work/vajira/DATA/FastGAN_polyps/generated_1k_set_3' --save_option "image_only"
python generate_4ch.py --dist '/work/vajira/DATA/FastGAN_polyps/generated_1k_set_4' --save_option "image_only"
'

for i in 5 10 15 20 25 30 35 40 45 50
do
    for s in 1 2 3 4
    do
        echo "working on $i ok..."
        python generate_4ch.py --ckpt "/work/vajira/DL/FastGAN-pytorch/train_results/test_4ch_num_img_${i}/models/all_50000.pth" --dist "/work/vajira/DATA/FastGAN_polyps/data_from_small_models_for_FID/gen_${i}_set_${s}" --save_option "image_only" --n_sample $i
    done
done
