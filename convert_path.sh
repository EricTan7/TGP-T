FILE_PATH=$1

for DATASET in caltech-101 dtd eurosat fgvc_aircraft food-101 imagenet stanford_cars oxford_pets oxford_flowers ucf101 sun397
do
    for SHOT in 1 2 4 8 16 
    do
        for SEED in 1 2 3
        do
            python tools/convert_path.py \
            --input-file ${FILE_PATH}/${DATASET}/split_fewshot/shot_${SHOT}-seed_${SEED}.pkl \
            --cvt-path /mnt/nas/TrueNas1/tanhao/recognition
        done
    done
done