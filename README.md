# singan-seg-polyp

## Install pip package
```shell
pip install singan-seg-polyp
```

## Import required packages
```python
from singan_seg_polyp import generate_data, prepare_requirements
```
---
## Prepare checkpoints
```python
Help on function prepare_checkpoints in module singan_seg_polyp.prepare_requirements:

prepare_checkpoints(path_to_checkpoints: str, link_keys=['link1', 'link2', 'link3', 'link4'], real_data=True, *args, **kwargs) -> str
'''
    The main function preparing checkpoints for pre-trained SinGANs of Polyp images.
    
    Parameters
    -----------
    path_to_checkpoints: str
        A directory path to download checkpoints. 
    link_keys: list
        A list of link keys: link1, link2, link3, link4. One or multiple link keys can be put in this list. 
    real_data: bool
        If True, the real images and masks used to train SinGANs will be downloaded to the checkpoint directory.  
    
    Return
    ------
    checkpoint_paths_list, real_image_mask_pair_list
        A sorted list of paths to downloaded checkpoints.
        A sorted (image_path, mask_path) tuple list.
'''
```

```python
out_paths = prepare_requirements.prepare_checkpoints("./singan_polyp_checkpoints", link_keys=["link1", "link2", "link3", "link4"])
```
### Parameters
- path_to_checkpoints -- A path to save checkpoints. The prepare_checkpoints() function checks the availability of pre-downloaded checkpoints. So, the use can run the save command twice without any downlaod overhead. 

- link_keys -- A list of pre-defined keys to download links. To download the full checkpoint list, use link_keys=["link1", "link2", "link3", "link4"]. If the user needs only a half of check points, then, the user can use only a half of link_keys. For example, link_keys= ["link1", "link2"]

- real_data -- If this is True, real images and masks used to train SinGANs will be downloaded into the checkpoint folder. 

### Return
This function returns a list of paths to all downloaded checkpoints to use with other functions of singan_polyp_aug. If the all checkpoints are used, then the function returns 1000 different sinGAN checkpoint paths which are dirrecting to pre-trained SinGAN checkpoints of the 1000 polyp images introduced in Hyper-kvasir dataset. 

----


## Generate synthetic polyps and corresponding mask


```python
generate_data.generate_from_single_checkpoint(out_dir:str, 
                                    checkpoint_path:str, 
                                    num_samples:int=1, 
                                    gen_start_scale:int=5,
                                    mask_post_processing:bool=True) -> None:
    ''' A function to generate synthetic polyp and correspondign mask from a given checkpoint path.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    checkpoint_path: str
        A path to a downloaded checkpoint. To get paths, you have to run prepare_requirements.prepare_checkpoints() function.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''
```

```python
generate_data.generate_from_multiple_checkpoints(out_dir:str, checkpoint_paths:list, *args, **kwargs)-> None:
    ''' A function to generate synthetic polyp and correspondign mask from a given list of checkpoint paths.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    checkpoint_paths: list
        A path list to downloaded checkpoints. To get paths, you have to run prepare_requirements.prepare_checkpoints() function.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''
```

```python
generate_data.generate_simple(out_dir:str, chk_dir:str, *args, **kwargs)-> None:
    ''' A function to generate synthetic polyp and correspondign mask from all downloaded checkpoint paths.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    chk_dir: str
        The path to checkpoint directory. If the directory does not have downloaded checkpoints, this function will download them.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''
```


## Transfering style from real to fake
```Python
help(style_transfer.transfer_style)

transfer_style(content_img_path: str, style_img_path: str, num_epochs: int, content_weight: int, style_weight: int, device: torch.device, vgg_model: str, verbose=False, tqdm_position=0, tqdm_leave=True, *args, **kwargs) -> 'PIL.Image' 
    Trnsfering style from a source image to a target image.
    
    Parameters
    ==========
    
    content_img_path: str
        A path to an image to which the style is going to be transfered.
    style_img: str
        A path to an image which has the required style to be transferred.
    num_epochs: int
        Number of epoch to iterate for transfering style to content image.
    content_weight: int
        Weight to keep the content of the destination image.
    style_weight: int
        Weight to transfer style from the source image.
    device: torch.device
        Torch device object, either "CPU" or "CUDA". Refer Pytoch documentation for more detials.
    vgg_model: str
        A model to extract features.
    verbose: bool
        If true, loss values will be printed to stdout.
    
    
    
    Return
    =======
    PIL.Image
        Style transferred image.
```



----
### Citation:
```
TBA
```

### References:
```

@article{cite-key,
	da = {2020/08/28},
	date-added = {2021-03-27 01:08:18 +0100},
	date-modified = {2021-03-27 01:08:18 +0100},
	doi = {10.1038/s41597-020-00622-y},
	id = {Borgli2020},
	isbn = {2052-4463},
	journal = {Scientific Data},
	number = {1},
	pages = {283},
	title = {HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy},
	ty = {JOUR},
	url = {https://doi.org/10.1038/s41597-020-00622-y},
	volume = {7},
	year = {2020},
	Bdsk-Url-1 = {https://doi.org/10.1038/s41597-020-00622-y}}

@inproceedings{shaham2019singan,
  title={Singan: Learning a generative model from a single natural image},
  author={Shaham, Tamar Rott and Dekel, Tali and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4570--4580},
  year={2019}
}

```

## Contacts:

### <vajira@simula.no> or <michael@simula.no>
