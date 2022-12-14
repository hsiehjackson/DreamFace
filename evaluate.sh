CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py \
         --ddim_eta 0.0 \
         --n_samples 1 \
         --n_iter 1 \
         --scale 7.0 \
         --ddim_steps 50 \
         --prompt "* person as supergirl, realistic, intricate, elegant, art by artgerm and wlo" \
         --prompt-subject "* person" \
         --prompt-origin  "person as supergirl, realistic, intricate, elegant, art by artgerm and wlo" \
         --outdir "outputs/" \
         --skip_grid \
         --ckpt "/data/chengping/dreambooth/baseline-star/user01/checkpoints/last.ckpt" \
         --seed 42