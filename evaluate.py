import os
import json


with open('/home/chengping/cse291/project/prompts_human.txt') as f:
    lines = f.readlines()
    lines = [eval(line.strip()) for line in lines]
    
for idx, p in enumerate(lines):
    # name = f'prompt{idx:02}'
    # prompt = p['new_prompt'].format(user=f'person')
    # command = f'CUDA_VISIBLE_DEVICES=1 python scripts/stable_txt2img.py \
    #  --ddim_eta 0.0 \
    #  --n_samples 8 \
    #  --n_iter 2 \
    #  --scale 7.0 \
    #  --ddim_steps 50 \
    #  --prompt "{prompt}" \
    #  --outdir "outputs/{name}" \
    #  --skip_grid \
    #  --ckpt "/data/chengping/ldm-ckpt/ori/sd-v1-4-full-ema.ckpt"'
    # os.system(command)
    
    for user in ['01', '02', '05', '09']:
        name = f'prompt{idx:02}-user{user}'
        prompt_sub = '* person'
        prompt_ori = p['new_prompt'].format(user='person')
        prompt = p['new_prompt'].format(user=prompt_sub)
        command = f'CUDA_VISIBLE_DEVICES=2 python scripts/stable_txt2img.py \
         --ddim_eta 0.0 \
         --n_samples 8 \
         --n_iter 2 \
         --scale 7.0 \
         --ddim_steps 50 \
         --prompt "{prompt}"  \
         --prompt-subject "{prompt_sub}" \
         --prompt-origin  "{prompt_ori}" \
         --outdir "outputs-personalize/{name}" \
         --skip_grid \
         --ckpt "/data/chengping/dreambooth/baseline-star/user{user}/checkpoints/last.ckpt"'
        os.system(command)