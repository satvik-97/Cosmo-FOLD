import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningDataModule



d = 64
shape = d
# tot_size = 9072
# tot_size = 267712
# tot_size =261242
# tot_size = 41160

# tot_size = int(8232*2)
tot_size = 6000

class AstroDataset(TensorDataset):
    def __init__(self, DM, HI,scale_halo,scale_tb):
        assert len(DM) == len(HI)
        self.DM = DM
        # self.HI = torch.log(torch.sum(HI,axis=1)).reshape(-1,1,32,32,32)
        self.HI = HI
        self.scale_halo = scale_halo
        self.scale_tb = scale_tb
    def __len__(self):
        return len(self.DM)

    def __getitem__(self, index):
        
        DM = (self.DM[index])/2.0  
        HI = (self.HI[index])/2.0   #/5
        # HI[:3]*=0.1
        return DM, HI #x,y
        # return DM, HI #x,y


class AstroDataModule(LightningDataModule):
    def __init__(
            self, 
            batch_size=8,
            num_workers=1,
            dataset='Sherwood'
        ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

    def setup(self, stage=None):

        if stage == "fit" or stage is None:

        

            # Tb_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/final_dataset/Aug_maps_32/64_rotate_dataset_tb_true.npy'
            # # Tb_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/final_dataset/Aug_maps_32/64_rotate_dataset_tb3.npy'
            # dm_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/final_test/Aug_maps_32/64_rotate_dataset_dm.npy'
            # halo_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/final_test/Aug_maps_32/64_rotate_dataset_halos1.npy' 
            # # halo_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/final_dataset/Aug_maps_32/64_rotate_dataset_concat_4bins_halos.npy'
            # # dm_file = '/data/samishr/CAMELS_25_256/Hydro/CV/1/Particle_dataset/Aug_maps_32/64_rotate_dataset_dm.npy'

            # Tb_file
            
            # Tb_file = '/data/samishr/CAMELS_25_256/Hydro/CV/train/CV_12345_tb.npy'
            # # # dm_file = '/data/samishr/CAMELS_25_256/custom_datasets/1_11_32_rotate_dm.npy'
            # halo_file = '/data/samishr/CAMELS_25_256/Hydro/CV/train/CV_12345_halos.npy'
            
            
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/DM/256/CV/delta_DM_01234_no_cutoff.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/Mstar/256/CV/delta_Mstar_01234.npy'
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/DM/256/CV/combined_dm_01234567_128_64.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/Mstar/256/CV/combined_mstar_01234567_128_64.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/T_gas/256/CV/T_01234.npy'
            # dm = np.memmap('/data/samishr/CAMELS_25_256/paper2/DM/256/CV/delta_DM_0134_no_cutoff_64.npy', dtype='float32', mode='r', shape=(tot_size*4,1,d,d,d))
            # mstar = np.memmap('/data/samishr/CAMELS_25_256/paper2/Mstar/256/CV/delta_Mstar_0134_64.npy', dtype='float32', mode='r', shape=(tot_size*4,1,d,d,d))
            # t = np.memmap('/data/samishr/CAMELS_25_256/paper2/T_gas/256/CV/T_0134_64.npy', dtype='float32', mode='r', shape=(tot_size*4,1,d,d,d))
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/DM/256/CV/delta_DM_0134_no_cutoff_64.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/T_gas/256/CV/T_0134_64.npy'
            ####### lowres
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/DM128/delta_DM_0_to_14-2-8_64.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/Mstar128/delta_Mstar_0_to_14-2-8_64.npy'
            
            ##### FASTPM 128 to TNG
            
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/FASTPM_128_TNG/true_dm_log_combined.npy'
            Tb_file = '/data/samishr/CAMELS_25_256/paper2/FASTPM_128_TNG/residual_combined.npy'
            halo_file = '/data/samishr/CAMELS_25_256/paper2/FASTPM_128_TNG/app_dm_log_combined.npy'
            
            #### TNG300
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/chunk_256_TNG/mstar_maps_64_pos_encoded_5freq_combined.npy'
            # # Tb_file = '/data/samishr/CAMELS_25_256/paper2/chunk_256_TNG/star_maps_64.npy'
            
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/chunk_256_TNG/dm_maps_1e-2_64_combined.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/chunk_256_TNG/T_maps_64.npy'
            
            ###### CAMELS to TNG
            # Tb = np.load('/data/samishr/CAMELS_25_256/Hydro/CV/CAMELS_for_TNG/star_128_1_4.npy')
            # halo = np.load('/data/samishr/CAMELS_25_256/Hydro/CV/CAMELS_for_TNG/dm_128_1_4.npy')
            
            #### wavelet
            
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/wave_data_TNG/hr_pred_4_diff_l2.npy'
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/wave_data_TNG/lr_cond_4_diff_l2.npy'
            
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/wave_data_TNG/hr_pred_4_l1_diff_16.npy'
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/wave_data_TNG/lr_cond_4_l1_diff_16.npy'
            

            # ####### DM 128, mstar 128
            # halo_file = '/data/samishr/CAMELS_25_256/paper2/DM/128/CV/delta_DM_0to7_64.npy'
            # Tb_file = '/data/samishr/CAMELS_25_256/paper2/Mstar/128/CV/delta_Mstar_0to7_64.npy'
            
            
            Tb = np.memmap(Tb_file, dtype='float32', mode='r', shape=(tot_size,1,d,d,d))
            halo = (np.memmap(halo_file, dtype='float32', mode='r', shape=(tot_size,1,d,d,d)))
            # Tb = Tb[:,1:]
            
            scale_halo = 1
            scale_tb = 1
            Halo = Tensor(halo)
            Tb = Tensor(Tb)

            # Halo = Tensor(Halo)
            # Tb = Tensor(Tb)


            print("SHAPE OF DATA",Halo.shape)
            # print("SHAPE OF VALIDATION DATA",Halo_valid.shape)
            generator = torch.Generator().manual_seed(34) #342
            # self.train_data = AstroDataset(Halo_train, Tb_train,scale)
            data = AstroDataset(Halo, Tb,scale_halo,scale_tb)
            # self.valid_data = AstroDataset(Halo_valid, Tb_valid,scale)
            train_set_size = int(len(Halo)*0.70)
            valid_set_size = int(len(Halo)-train_set_size)
            self.train_data, self.valid_data = random_split(
                data, [train_set_size, valid_set_size], generator=generator
            )
            # print("LENGTH OF DATA",train_set_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_data, batch_size=1, num_workers=self.num_workers
    #     )



def get_astro_data(dataset, num_workers=1, batch_size=10, stage=None):

    dm = AstroDataModule(
        num_workers=num_workers,
        batch_size=batch_size,
    )
    dm.setup(stage=stage)
    return dm