import urllib, os
from tqdm import tqdm
import requests
from zipfile import ZipFile
import time
import yaml
import sys
import shutil
from pathlib import Path
import numpy as np
from natsort import natsorted

def load_configs()->dict:
    """ Loading the download paths in config file

    Parameters
    -----------
    No input parameters.

    Return
    -------
    dict
        A dictionary containing all the download links. 

    """
    
    print(os.getcwd())
    module_path = str(Path(__file__).parent.absolute())
    print(module_path)
    
    with open(os.path.join(module_path, "config.yaml")) as f:
        output = yaml.safe_load(f)
    f.close()
    
    return dict(output)


def extract_zip_file(zip_path:str, dst_dir:str):
    
    print("=== Extracting files ===")
    time.sleep(2)
    with ZipFile(file=zip_path) as zip_file:

        # Loop over each file
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):

            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_file.extract(member=file, path=dst_dir)



def download_and_extract_single_file(url:str, path_to_extract:str, extracting:bool = True, clean:bool =False):
    
    response = getattr(urllib, 'request', urllib).urlopen(url)
    
    filesize = int(response.headers.get('content-length', 0))#int(requests.head(url).headers["Content-Length"])
    chunk_size = 1024
    
    filename = os.path.basename(url).split("?")[0]
    os.makedirs(path_to_extract, exist_ok=True)
    
    abs_path = os.path.join(path_to_extract, filename)
    
    directory = abs_path[:-4]
    
    #print(directory)
    
    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"The directory:{directory} is already exists.")
        return directory
    
    elif os.path.exists(abs_path):
        print(f"The zip file: {abs_path} is already exists.")
        
        if extracting:
            print("Extracting TRUE...!")
            extract_zip_file(abs_path, path_to_extract)
        return directory
    
    else:
    
   
        with requests.get(url, stream=True) as r, open(abs_path, "wb") as f, tqdm(
            unit="B",  # unit string to be displayed.
            unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
            unit_divisor=1024,  # is used when unit_scale is true
            total=filesize,  # the total iteration.
            file=sys.stdout,  # default goes to stderr, this is the display on console.
            desc=filename  # prefix to be displayed on progress bar.
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                # download the file chunk by chunk
                datasize = f.write(chunk)
                # on each chunk update the progress bar.
                progress.update(datasize)
        f.close()
        
        if extracting:
            print("Extracting TRUE...!")
            extract_zip_file(abs_path, path_to_extract)
        
    if clean:
        os.remove(abs_path)
        
    return directory


def prepare_checkpoints(path_to_checkpoints:str, link_keys=["link1","link2","link3","link4"], real_data=True,*args, **kwargs)-> str:
    """ The main function preparing checkpoints for pre-trained SinGANs of Polyp images.

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
    """
    
    all_links = load_configs()["links"]

    real_data_links = load_configs()["real_data_links"]
    
    #alls_in_one_dir = os.path.join(path_to_checkpoints, "all")
    #os.makedirs(alls_in_one_dir, exist_ok=True)
    
    checkpoint_paths = []
    
    for link_key in link_keys:
        print(all_links[link_key])
        download_link = all_links[link_key]
        
        directory = download_and_extract_single_file(download_link, path_to_checkpoints)
        #print("Directory=", directory)
        
        checkpoint_paths = checkpoint_paths + list(Path(directory).iterdir())
        
        ## moving checkpoints to root directory
        #for sub_dir in tqdm(Path(directory).iterdir()):
            #print(sub_dir)
        #    shutil.move(str(sub_dir), alls_in_one_dir)
    checkpoint_paths_str = [str(p) for p in checkpoint_paths]

    # Download and prepair real images and maks
    real_data_paths = None
    if real_data:
        image_directory = download_and_extract_single_file(real_data_links["images_link"], path_to_checkpoints)
        mask_directory = download_and_extract_single_file(real_data_links["masks_link"], path_to_checkpoints)

        image_paths = list(Path(image_directory).iterdir())
        mask_paths = list(Path(mask_directory).iterdir())

        image_paths = [str(p) for p in image_paths]
        mask_paths = [str(p) for p in mask_paths]

        image_paths = natsorted(image_paths)
        mask_paths = natsorted(mask_paths)

        real_data_paths = list(zip(image_paths, mask_paths))


    return natsorted(checkpoint_paths_str), real_data_paths