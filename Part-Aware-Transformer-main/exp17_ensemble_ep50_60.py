import numpy as np
qf50 = np.load('qf_exp17_ep50.npy')
gf50 = np.load('gf_exp17_ep50.npy')
qf60 = np.load('qf_exp17_ep60.npy')
gf60 = np.load('gf_exp17_ep60.npy')
qf_ens = qf50 + qf60
qf_ens = qf_ens / np.linalg.norm(qf_ens, axis=1, keepdims=True)
gf_ens = gf50 + gf60
gf_ens = gf_ens / np.linalg.norm(gf_ens, axis=1, keepdims=True)
np.save('qf_ensemble_exp17_ep50_60.npy', qf_ens)
np.save('gf_ensemble_exp17_ep50_60.npy', gf_ens)
print('Done. qf:', qf_ens.shape, 'gf:', gf_ens.shape)