# GAVE FAR WORSE RESULTS
import numpy as np

# exp13 ep30+ep40 ensemble
qf13 = np.load('qf_ensemble_ep30_ep40.npy')
gf13 = np.load('gf_ensemble_ep30_ep40.npy')

# exp15 ep50+ep55 ensemble
qf15 = np.load('qf_ensemble_exp15_50_55.npy')
gf15 = np.load('gf_ensemble_exp15_50_55.npy')

print('exp13 qf:', qf13.shape, 'exp15 qf:', qf15.shape)

# Average and L2 normalize
qf_cross = qf13 + qf15
qf_cross = qf_cross / np.linalg.norm(qf_cross, axis=1, keepdims=True)
gf_cross = gf13 + gf15
gf_cross = gf_cross / np.linalg.norm(gf_cross, axis=1, keepdims=True)

np.save('qf_cross_exp13_exp15.npy', qf_cross)
np.save('gf_cross_exp13_exp15.npy', gf_cross)
print('Done. qf_cross:', qf_cross.shape, 'gf_cross:', gf_cross.shape)