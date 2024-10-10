# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:22:37 2024

@author: Dragon
"""

import torch
from monai.transforms import Transform

class EqualProbabilityRandFlipd(Transform):
    def __init__(self, keys, spatial_axis=(1, 2, 3)):
        self.keys = keys
        self.spatial_axis = spatial_axis
        self.num_combinations = 2 ** len(spatial_axis)

    def __call__(self, data):
        flip_choice = torch.randint(self.num_combinations, (1,)).item()
        flip_axes = [axis for i, axis in enumerate(self.spatial_axis) if flip_choice & (1 << i)]
        
        for key in self.keys:
            if torch.is_tensor(data[key]):
                data[key] = torch.flip(data[key], dims=flip_axes)
            else:
                raise TypeError(f"Data with key {key} is not a torch.Tensor")
        
        return data
    
def apply_noise(data_dict, noise, gauss):
    image = data_dict['image']
    label = data_dict['label']
    
    # Generate noise factors
    noiseA = torch.randn(1, dtype=image.dtype) * noise
    noiseB = torch.randn(1, dtype=image.dtype) * noise

    # Apply noise
    image = image * ( 1 + noiseA) + noiseB

    # Add Gaussian noise
    gaussian_noise = torch.randn(image.shape, dtype=image.dtype) * gauss
    image = image + gaussian_noise
    
    new_dict = {
        'image': image,
        'label': label,
    }
    
    return new_dict

# create empty dataframe for cumulative values
def getCumulativeAry(df):
    cumulativeary = np.zeros([len(df),7])

    for i in range(2,9):
        c=0
        # calculate cumulative values
        cumulative = 0
        for index, row in df.iterrows():
            cumulative += row[i]
            cumulativeary[c,i-2]=int(cumulative)
            c+=1
        # combine original and cumulative dataframes
        #df = pd.concat([df, cumulative_df], axis=1)

    cumulativeary=cumulativeary.astype(int)
    # print result
    return cumulativeary