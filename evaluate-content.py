import os
import numpy as np
from deepface import DeepFace

genBaseDir = '/home/chengping/Dreambooth-Stable-Diffusion/outputs-personalize'
# genBaseDir = '/home/chengping/textual_inversion/outputs'
contentBaseDir = '/home/chengping/Dreambooth-Stable-Diffusion/celebA/images'

total_content = []

for user in [1,2,5,9]:
    genImages = []
    for prompt in range(20):
        gendir = os.path.join(genBaseDir, f'prompt{prompt:02}-user{user:02}', 'samples')
        genImages += [os.path.join(gendir, i) for i in os.listdir(gendir) if i[-1] =='g']
    
    contentDir = os.path.join(contentBaseDir, f'user{user:02}')
    contentImages = [os.path.join(contentDir, i) for i in os.listdir(contentDir) if i[-1] =='g']
    
    similarity = []
    for g in genImages:
        s = 0
        for c in contentImages:
            s += 1 - DeepFace.verify(img1_path=g, img2_path=c, enforce_detection=False, prog_bar=False)["distance"]
        similarity.append(s / len(contentImages))
    
    total_content.append(np.mean(similarity))
    
print(np.mean(total_content))