for DATASET in caltech-101 dtd eurosat fgvc_aircraft food-101 imagenet stanford_cars oxford_pets oxford_flowers ucf101 sun397
do
    for SEED in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/configs/${DATASET}.yaml \
        --seed ${SEED} \
        --use-wandb
    done
done
