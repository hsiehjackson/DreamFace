max_training_steps=2000
class_word="person"
reg_data_root="/home/chengping/Dreambooth-Stable-Diffusion/regularization_images/person_ddim" 

for id in "05" "09"
do
    user_token=user$id
    data_root=/home/chengping/Dreambooth-Stable-Diffusion/celebA/images/$user_token
    rm -rf $data_root/.ipynb_checkpoints
    rm -rf $reg_data_root/.ipynb_checkpoints
    CUDA_VISIBLE_DEVICES=3 python main.py \
     --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
     --actual_resume "/data/chengping/ldm-ckpt/ori/sd-v1-4-full-ema.ckpt" \
     --reg_data_root $reg_data_root \
     --gpus 0, \
     --data_root $data_root \
     --max_training_steps $max_training_steps \
     --class_word $class_word \
     --token '*' \
     --no-test \
     -t
done
