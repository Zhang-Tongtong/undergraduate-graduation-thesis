import numpy as np
import glob

files = sorted(glob.glob(
"experiments/Local_Module/FineDance_FineTuneV2_Local/samples_dod*/dod_*198*.npy"
))

motions = []

for f in files:
    motions.append(np.load(f))

dance = np.concatenate(motions,axis=0)

print(dance.shape)

np.save("dance.npy",dance)