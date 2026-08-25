import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from matplotlib.patches import Circle

import vdm_3d_model, vdm_3d_network
# from utils import compute_pk
# from data import constants

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

plt.style.use(['science', 'vibrant'])
mpl.rcParams['figure.dpi'] = 250
plt.rcParams.update({
    'axes.titlesize': 11,
    'legend.loc': 'upper right',
    'legend.fontsize': 6
    })


# dm = np.load('/data/samishr/CAMELS_25_256/paper2/merged/dm_processed.npy')
mstar = np.load('/data/samishr/CAMELS_25_256/paper2/merged/star_processed.npy')
# T = np.load('/data/samishr/CAMELS_25_256/paper2/merged/T_processed.npy')



def load_model(dataset = 'Astrid',
        cropsize = 64,
        gamma_min = -13.3,
        gamma_max = 13.3,
        embedding_dim = 48,
        norm_groups = 8,
        conditioning_channels = 1,
        use_fourier_features = False,
        add_attention = True,
        noise_schedule = 'learned_linear',
        n_blocks = 4
):
    vdm = vdm_3d_model.LightVDM(
            score_model=vdm_3d_network.UNetVDM(
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                embedding_dim=embedding_dim,
                norm_groups=norm_groups,
                n_blocks=n_blocks,
                conditioning_channels=conditioning_channels,
                add_attention=add_attention,
                use_fourier_features=use_fourier_features
            ),
            dataset=dataset,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            image_shape=(1,cropsize,cropsize,cropsize),
            noise_schedule=noise_schedule,
        )
    vdm = vdm.to(device=device)
    vdm = vdm.eval()
    # ckpt = '/data/samishr/CAMELS_25_256/paper2/TNG/DM_Mstar/epoch=134-step=111240-val_loss=-0.8745.ckpt'
    ckpt = '/data/samishr/CAMELS_25_256/paper2/TNG/CAMELS_TNG_8_48/works_epoch_237.ckpt'
    
    # ckpt = '/data/samishr/CAMELS_25_256/paper2/TNG/DM_mstar_400/epoch=116-step=96408-val_loss=-1.0735.ckpt'
    # ckpt = '/data/samishr/CAMELS_25_256/paper2/DM_mstar_lowres/CAMELS_CV_1234_DM_from_mstar_8_48_64_extended_circular_t_400/epoch=204-step=212585-val_loss=-2.8567.ckpt'
    state_dict=torch.load(ckpt)["state_dict"]
    vdm.load_state_dict(state_dict)
    return vdm.eval()



def extract_chunks_reshape(volume, chunk_size=64):
    # volume shape: (1024,1024,1024)
    # First, reshape into (4,64,4,64,4,64)
    temp = volume.reshape(16, chunk_size, 16, chunk_size, 16, chunk_size)
    # Transpose to bring all chunk indices together: (16,16,16,64,64,64)
    temp = temp.permute(0, 2, 4, 1, 3, 5)
    # Flatten the first three dimensions to get 64 chunks:
    chunks = temp.reshape(-1, 1, chunk_size, chunk_size, chunk_size)
    return chunks


import numpy as np

def reconstruct_volume_from_chunks(chunks, chunk_size=64):
    """
    Reconstructs a 1024^3 volume from non-overlapping chunks.
    
    Parameters:
      chunks (np.ndarray): Array of shape (4096, 64, 64, 64) where each chunk is of shape (64, 64, 64).
      chunk_size (int): Size of each chunk along one dimension (default is 64).
      
    Returns:
      np.ndarray: Reconstructed volume of shape (256, 256, 256).
    """
    temp = chunks.reshape(16, 16, 16, chunk_size, chunk_size, chunk_size)
    # Rearrange axes to combine the chunk indices with the chunk content.
    temp = temp.permute(0, 3, 1, 4, 2, 5)
    # Finally, reshape to (256,256,256)
    volume = temp.reshape(16 * chunk_size, 16 * chunk_size, 16 * chunk_size)
    return volume



import gc 
# scale_mstar = 5.0
scale_mstar = 2.0 # only camels tng
scale_dm = 2.0



vdm = load_model()
n=1
@torch.no_grad()
def repaint_interleaved_lin_semi(mstar, device='cuda'):
    output = np.zeros((n,1024,1024,1024),dtype=np.float32)
    SEED = torch.tensor([1,2,3])
    n_sampling_steps = 100
    steps = torch.linspace(
            1.0,
            0.0,
            n_sampling_steps + 1,
            device='cpu',
        )
    for iter in range(n):
        torch.manual_seed(SEED[iter])
        # initialize once per iter
        updated_chunks = torch.randn((4096,1,64,64,64), device=device)
        current_vol = reconstruct_volume_from_chunks(updated_chunks).detach() # initial noisy 1024^3
        rolled_cond = torch.tensor(mstar/scale_mstar, device=device).detach() # initial conditional 1024^3

        for t_idx in range(n_sampling_steps):
            # pick chunks & rolled_cond
            if t_idx > 200:
                chunks = extract_chunks_reshape(current_vol) # noisy 64^3
                rolled_cond_chunks = extract_chunks_reshape(rolled_cond) # cond 64^3
            else:
                rolled_vol = torch.roll(current_vol, shifts=(1,1,1), dims=(-3,-2,-1)) # noisy roll
                chunks = extract_chunks_reshape(rolled_vol) # chunks roll noise
                rolled_cond = torch.roll(rolled_cond, shifts=(1,1,1), dims=(-3,-2,-1)).detach() # cond roll 
                rolled_cond_chunks = extract_chunks_reshape(rolled_cond) # chunks cond roll 

            # sample all sub‐chunks in‐place
            for samp in range(512):
                out = vdm.model.sample_zs_given_zt(
                    zt=chunks[samp*8:(samp+1)*8],
                    conditioning=rolled_cond_chunks[samp*8:(samp+1)*8],
                    t=steps[t_idx],
                    s=steps[t_idx+1]
                )
                updated_chunks[samp*8:(samp+1)*8].copy_(out)
                del out
                gc.collect()
                torch.cuda.empty_cache()
                print(samp)

            # reconstruct & detach
            current_vol = reconstruct_volume_from_chunks(updated_chunks).detach()

            # clean up before next t_idx
            del chunks, rolled_cond_chunks
            gc.collect()
            torch.cuda.empty_cache()
            print('step : ',t_idx)
        # save and move on
        output[iter] = current_vol.cpu().reshape(1024,1024,1024)
    return output


output200 = repaint_interleaved_lin_semi(mstar)

np.save('/data/samishr/CAMELS_25_256/paper2/TNG/DM_Mstar/TNG_output/CAMELS_TNG',output200)

print('saved')
