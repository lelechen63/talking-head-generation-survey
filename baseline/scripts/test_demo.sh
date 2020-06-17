test_model_vox_new_noani(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 4 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref \
    --use_new
}

test_model_vox_origin(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 4 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 8 \
    --serial_batches \
    --no_rt \
    --dataset_name vox
}

# test_model_vox_new_noani 3 face8_vox_new_baseline latest 10
test_model_vox_origin 3 face8_vox_new_baseline latest 10