test_model_vox_new_noani_2(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_frame.py --name $2 \
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
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref
}

test_model_lrs(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_frame.py --name $2 \
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
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrs \
    --crop_ref
}

test_model_lrs_wang(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_frame.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 4 \
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrs
}

test_model_lrw(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_frame.py --name $2 \
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
    --dataroot '/home/cxu-serve/p1/common/lrw' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrw \
    --crop_ref
}


# test_model_vox_new_noani_2 2 face8_vox_ani_nonlinear_noani 5 2000
# test_model_lrs 3 face8_lrs_nonlinear_noani latest 2000
# test_model_lrw 2 face8_lrw_nonlinear_noani latest 2000
test_model_lrs_wang 2 face8_lrs_wang latest 2000