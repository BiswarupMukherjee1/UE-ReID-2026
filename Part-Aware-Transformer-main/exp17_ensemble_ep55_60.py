import numpy as np
qf55 = np.load('qf_exp17_ep55.npy')
gf55 = np.load('gf_exp17_ep55.npy')
qf60 = np.load('qf_exp17_ep60.npy')
gf60 = np.load('gf_exp17_ep60.npy')
qf_ens = qf55 + qf60
qf_ens = qf_ens / np.linalg.norm(qf_ens, axis=1, keepdims=True)
gf_ens = gf55 + gf60
gf_ens = gf_ens / np.linalg.norm(gf_ens, axis=1, keepdims=True)
np.save('qf_ensemble_exp17_ep55_60.npy', qf_ens)
np.save('gf_ensemble_exp17_ep55_60.npy', gf_ens)
print('Done. qf:', qf_ens.shape, 'gf:', gf_ens.shape)