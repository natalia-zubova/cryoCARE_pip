#!/software/fg24661/miniconda/envs/cryocare_11/bin/python3.8
import argparse
import json
from os.path import join
import os
import tarfile
import tempfile
import datetime
import mrcfile
import numpy as np
import sys
import tensorflow as tf
from typing import Tuple

from cryocare.internals.CryoCARE import CryoCARE
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule
from csbdeep.data import NoResizer

import psutil

def set_gpu_id(config: dict):
    if 'gpu_id' in config:
        if type(config['gpu_id']) is list:
            gpu_ids = config['gpu_id']
            if len(gpu_ids) == 0:
                raise RuntimeError('ERROR: List of GPU IDs is empty')
        elif type(config['gpu_id']) is int:
            gpu_ids = [config['gpu_id']]
        else:
            raise RuntimeError('gpu_id in json is neither a list nor an integer')
    else:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            gpu_ids = list(range(0,len(tf.config.list_physical_devices('GPU'))))
        else:
            print('WARNING: No GPUs found by tensorflow')
    
    #Check GPUs given by IDs exist and set_memory_growth to True
    physical_devices = []
    try:
        for gpu in gpu_ids:
            print(f'Looking for GPU with ID: {gpu}')
            physical_devices = physical_devices + [tf.config.list_physical_devices('GPU')[gpu]]
            print(f'GPU {gpu} successfully found')
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[gpu], True)
    except IndexError:
        print(f'WARNING: GPU {gpu} not found')
    
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices, 'GPU') 

def pad(volume: np.array, div_by: Tuple) -> np.array:
    pads = []
    for axis_index, axis_size in enumerate(volume.shape):
        pad_by = axis_size%div_by[axis_index]
        pads.append([0,pad_by])
    volume_padded = np.pad(volume, pads, mode='mean')

    return volume_padded



def denoise(config: dict, mean: float, std: float, even: str, odd: str, output_file: str):
    model = CryoCARE(None, config['model_name'], basedir=config['path'])

    even = mrcfile.mmap(even, mode='r', permissive=True)
    odd = mrcfile.mmap(odd, mode='r', permissive=True)
    shape_before_pad = even.data.shape
    even_vol = even.data
    odd_vol = odd.data
    even_vol = even_vol
    odd_vol = odd_vol

    div_by = model._axes_div_by('XYZ')

    even_vol = pad(even_vol,div_by=div_by)
    odd_vol = pad(odd_vol, div_by=div_by)

    denoised = np.zeros(even_vol.shape)

    # Add channel dimension to arrays (in-place, consistent with original code)
    even_vol.shape += (1,)
    odd_vol.shape += (1,)
    denoised.shape += (1,)

    # prepare cropped views for the model (the model expects cropped inputs as in CryoCARE.predict)
    even_c = model._crop(even_vol)
    odd_c = model._crop(odd_vol)
    denoised_c = model._crop(denoised)

    # create containers for even/odd component predictions (cropped views)
    even_comp = np.zeros_like(denoised)
    odd_comp = np.zeros_like(denoised)
    even_comp_c = model._crop(even_comp)
    odd_comp_c = model._crop(odd_comp)

    # run prediction filling average into denoised_c and components into even_comp_c / odd_comp_c
    # use NoResizer to match CryoCARE.predict behavior
    model._predict_mean_and_scale(even_c, odd_c, denoised_c, axes='ZYXC', normalizer=None, resizer=NoResizer(),
                                  mean=mean, std=std, n_tiles=config['n_tiles'] + [1, ],
                                  even_out=even_comp_c, odd_out=odd_comp_c)

    # crop padded arrays back to original shape (may still have a singleton channel axis)
    denoised_final = denoised[slice(0, shape_before_pad[0]), slice(0, shape_before_pad[1]), slice(0, shape_before_pad[2])]
    even_final = even_comp[slice(0, shape_before_pad[0]), slice(0, shape_before_pad[1]), slice(0, shape_before_pad[2])]
    odd_final = odd_comp[slice(0, shape_before_pad[0]), slice(0, shape_before_pad[1]), slice(0, shape_before_pad[2])]

    # remove trailing singleton channel axis if present so shapes match the original tomogram
    def _squeeze_channel_if_needed(arr, target_shape):
        if arr.shape == target_shape:
            return arr
        # if last axis is singleton and squeezing yields target shape, do it
        if arr.ndim == len(target_shape) + 1 and arr.shape[-1] == 1 and arr[..., 0].shape == target_shape:
            return arr[..., 0]
        return arr

    denoised_final = _squeeze_channel_if_needed(denoised_final, shape_before_pad)
    even_final = _squeeze_channel_if_needed(even_final, shape_before_pad)
    odd_final = _squeeze_channel_if_needed(odd_final, shape_before_pad)

    # sanity check shapes
    if denoised_final.shape != shape_before_pad:
        raise RuntimeError(f"Averaged denoised volume has incorrect shape {denoised_final.shape}, expected {shape_before_pad}")
    if even_final.shape != shape_before_pad:
        raise RuntimeError(f"Even denoised volume has incorrect shape {even_final.shape}, expected {shape_before_pad}")
    if odd_final.shape != shape_before_pad:
        raise RuntimeError(f"Odd denoised volume has incorrect shape {odd_final.shape}, expected {shape_before_pad}")

    # Verify that (even + odd) / 2 reconstructs the averaged prediction within tolerance
    avg = denoised_final.astype(np.float64)
    recon = (even_final.astype(np.float64) + odd_final.astype(np.float64)) / 2.0
    max_abs = float(np.max(np.abs(avg - recon)))
    mean_abs = float(np.mean(np.abs(avg - recon)))
    print(f"Component reconstruction check: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}")
    if not np.allclose(avg, recon, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"Even/odd components do not reconstruct averaged output (max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e})")

    # Write averaged volume (same behavior as original)
    mrc = mrcfile.new_mmap(output_file, denoised_final.shape, mrc_mode=2, overwrite=True)
    mrc.data[:] = denoised_final

    # Write even and odd volumes to separate files next to averaged output
    base, ext = os.path.splitext(output_file)
    even_out_file = base + '_even' + ext
    odd_out_file = base + '_odd' + ext

    mrc_even = mrcfile.new_mmap(even_out_file, even_final.shape, mrc_mode=2, overwrite=True)
    mrc_even.data[:] = even_final

    mrc_odd = mrcfile.new_mmap(odd_out_file, odd_final.shape, mrc_mode=2, overwrite=True)
    mrc_odd.data[:] = odd_final





    for l in even.header.dtype.names:
        if l == 'label':
            new_label = np.concatenate((even.header[l][1:-1], np.array([
                'cryoCARE                                                ' + datetime.datetime.now().strftime(
                    "%d-%b-%y  %H:%M:%S") + "     "]),
                                        np.array([''])))
            print(new_label)
            mrc.header[l] = new_label
        else:
            mrc.header[l] = even.header[l]
    mrc.header['mode'] = 2
    mrc.set_extended_header(even.extended_header)
    # copy header for even/odd outputs as well
    try:
        for l in even.header.dtype.names:
            if l == 'label':
                new_label = np.concatenate((even.header[l][1:-1], np.array([
                    'cryoCARE                                                ' + datetime.datetime.now().strftime(
                        "%d-%b-%y  %H:%M:%S") + "     "]),
                                                np.array([''])))
                mrc_even.header[l] = new_label
                mrc_odd.header[l] = new_label
            else:
                mrc_even.header[l] = even.header[l]
                mrc_odd.header[l] = even.header[l]
        mrc_even.header['mode'] = 2
        mrc_odd.header['mode'] = 2
        mrc_even.set_extended_header(even.extended_header)
        mrc_odd.set_extended_header(even.extended_header)
    except Exception:
        # non-fatal: header copy failed
        pass

def main():
    
    parser = argparse.ArgumentParser(description='Run cryoCARE prediction.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    try:
        os.makedirs(config['output'])
    except OSError:
        if 'overwrite' in config and config['overwrite']:
            os.makedirs(config['output'], exist_ok=True)
        else:
            print("Output directory already exists. Please choose a new output directory or set 'overwrite' to 'true' in your configuration file.")
            sys.exit(1)
    
    set_gpu_id(config)
    
    if os.path.isfile(config['path']):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tar = tarfile.open(config['path'], "r:gz")
            tar.extractall(tmpdirname)
            tar.close()
            config['model_name'] = os.listdir(tmpdirname)[0]
            config['path'] = os.path.join(tmpdirname)
            with open(os.path.join(tmpdirname,config['model_name'],"norm.json")) as f:
                norm_data = json.load(f)
                mean = norm_data["mean"]
                std = norm_data["std"]



            from glob import glob
            if type(config['even']) is list:
                all_even=tuple(config['even'])
                all_odd=tuple(config['odd'])
            elif os.path.isdir(config['even']) and os.path.isdir(config['odd']):
                all_even = glob(os.path.join(config['even'],"*.mrc"))
                all_odd = glob(os.path.join(config['odd'],"*.mrc"))
            else:
                all_even = [config['even']]
                all_odd = [config['odd']]

            for even,odd in zip(all_even,all_odd):
                out_filename = os.path.join(config['output'], os.path.basename(even))
                denoise(config, mean, std, even=even, odd=odd, output_file=out_filename)
    else:
        # Fall back to original cryoCARE implmentation
        s = f" {config['path']} is not a file"
        if os.path.exists(config['path']):
            s = f" {config['path']} does not exist"
        print(f"The specified 'path' {s}. Your config is not in the format that cryoCARE >=0.2 requires. Fallback to cryCARE 0.1 format.")
        if 'output_name' not in config or os.path.isfile(config['path']):
            print("Invalid config format.")
            sys.exit(1)

        dm = CryoCARE_DataModule()
        dm.load(config['path'])
        mean, std = dm.train_dataset.mean, dm.train_dataset.std

        denoise(config, mean, std, even=config['even'], odd=config['odd'], output_file=join(config['path'], config['output_name']))




if __name__ == "__main__":
    main()
