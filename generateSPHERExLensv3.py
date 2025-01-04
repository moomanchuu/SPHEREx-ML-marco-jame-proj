#Final Script to Run to generate the data needed to train the model: 

import multiprocessing
from tqdm import tqdm
import numpy as np
import h5py
from pathlib import Path
import sys
import copy
import os

# Setup paths for imports
base_dir = Path.cwd()
autolens_config_path = base_dir / "autolens_config"
spherex_path = base_dir / "SPHEREx"

sys.path.append(str(autolens_config_path))
sys.path.append(str(spherex_path))

import SPHERExScripts as SPHEREx  # SPHEREx module
import gcluster15 as galaxy  # Galaxy generating functions


def generate_and_process(index, probLensing=1):
    """
    Generate a galaxy cluster image, process it to SPHEREx resolution, and return results.

    Parameters:
        index (int): The unique index for the current process.
        probLensing (float): Probability of generating a lensed object (0 to 1).

    Returns:
        dict: Contains index, lensing flag, and processed arrays.
    """
    np.random.seed(int.from_bytes(os.urandom(8), byteorder='little')) #Change the RNG seed

    # Randomly determine lensing based on the probability
    lensing = 1 if np.random.random() < probLensing else 0

    rawImage = galaxy.wrapperFunction(
        n_cluster_members=5,
        cluster_central_redshift=0.5,
        source_redshift=1.5,
        lensing=lensing,
        plot_result=False
    )

    darkCurrentRate = 0.02  # e-/s/pixel (from: https://ar5iv.labs.arxiv.org/html/1412.4872)
    integrationTime = 5  # seconds, this is prob pseudo-completely arbitary? 
    noiseParameter = darkCurrentRate*integrationTime #note, larger parameter = more noise

    raw = copy.deepcopy(np.array(rawImage.data.native))

    convolved, binned = SPHEREx.processImg(copy.deepcopy(raw))
    darkCurrentArray = SPHEREx.addNoise(binned,mode="constantdarkcurrent", parameter = noiseParameter)

    return {
        "index": index,
        "isLensed": lensing,
        "simulated_array": raw,
        "SPHEREx_Array_binned": binned,
        "SPHEREx_Array_dc": darkCurrentArray
    }


def save_results(results, output_path):
    """
    Save generated and processed images into an HDF5 file.
    """
    with h5py.File(output_path, "w") as hdf:
        indices = [result["index"] for result in results]
        is_lensed = [result["isLensed"] for result in results]
        hdf.create_dataset("indices", data=np.array(indices, dtype=int))
        hdf.create_dataset("isLensed", data=np.array(is_lensed, dtype=int))

        simulated_shape = results[0]["simulated_array"].shape
        spherex_shape = results[0]["SPHEREx_Array_binned"].shape


        simulated_dataset = hdf.create_dataset(
            "simulated_arrays",
            shape=(len(results), *simulated_shape),
            dtype="f"
        )
        spherexBinned_dataset = hdf.create_dataset(
            "SPHEREx_Array_binned",
            shape=(len(results), *spherex_shape),
            dtype="f"
        )

        spherexDC_dataset = hdf.create_dataset(
            "SPHEREx_Array_dc",
            shape=(len(results), *spherex_shape),
            dtype="f"
        )

        for i, result in enumerate(results):
            simulated_dataset[i] = result["simulated_array"]
            spherexBinned_dataset[i] = result["SPHEREx_Array_binned"]
            spherexDC_dataset[i] = result["SPHEREx_Array_dc"]

    print(f"Results successfully saved to {output_path}")


def task_wrapper(args):
    """
    Wrapper function to allow multiprocessing with arguments.

    Parameters:
        args (tuple): Arguments passed to generate_and_process.
    """
    return generate_and_process(*args)


def main(num_images, probLensing=1, output_path="output.h5"):
    """
    Main function to generate and process galaxy images in parallel.
    """
    # Single-threaded version for debugging
    # results = [
    #     generate_and_process(index, probLensing=probLensing)
    #     for index in tqdm(range(num_images), desc="Processing images")
    # ]
    
    args = [(index, probLensing) for index in range(num_images)]

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap(task_wrapper, args),
                total=num_images,
                desc="Processing images"
            )
        )
    
    save_results(results, output_path)

    return results


if __name__ == "__main__":
    num_images = 5  # Prob need 500+ lmao 
    probLensing = 0.5  # 1 = only lensing objects, 0 = no lensing, 0.5 = 50-50
    output_path = "output/test5.h5"  # Path to save the output file

    results = main(num_images, probLensing, output_path)



















'''
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import autolens_config as al
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

base_dir = Path.cwd()

autolens_config_path = base_dir / "autolens_config"
spherex_path = base_dir / "SPHEREx"

sys.path.append(str(autolens_config_path))
sys.path.append(str(spherex_path))


import SPHERExScripts as SPHEREx  # from SPHEREX
import gcluster15 as galaxy #galaxy generating function



def generate_and_process(index, probLensing=1):
    """
    Generate a lensed galaxy cluster image, process it to SPHEREx resolution, and return the results.

    Parameters:
        probLensing (float): Probability of lensing occurring (0 to 1).
        index (int): Index of the process for tracking.

    Returns:
        dict: A dictionary containing the raw and processed images and their numpy arrays.
    """

    # Randomize lensing based on the probability parameter
    lensing = 1 if np.random.random() < probLensing else 0

    rawImage = galaxy.wrapperFunction(
        n_cluster_members=5,
        cluster_central_redshift=0.5,
        source_redshift=1.5,
        lensing=lensing,
        plot_result=False
    )

    raw = np.array(rawImage.data.native) # Convert rawImage to 2D Numpy Array

    convolved, binned = SPHEREx.processImg(raw) # Convert rawImage to 2D Numpy Array

    return {
        "index": index,
        "isLensed": lensing,
        "simulated_array": raw,
        "spherex_array": binned
    }

import h5py

def save_results(results, output_path):
    """
    Save the generated and processed images into an HDF5 file.
    """
    # Open an HDF5 file for writing
    with h5py.File(output_path, "w") as hdf:
        # Prepare datasets for indices and lensing flags
        indices = [result["index"] for result in results]
        is_lensed = [result["isLensed"] for result in results]
        hdf.create_dataset("indices", data=np.array(indices, dtype=int))
        hdf.create_dataset("isLensed", data=np.array(is_lensed, dtype=int))

        # Determine the shape of 2D arrays
        simulated_shape = results[0]["simulated_array"].shape
        spherex_shape = results[0]["spherex_array"].shape

        # Create datasets for simulated and SPHEREx arrays
        simulated_dataset = hdf.create_dataset(
            "simulated_arrays",
            shape=(len(results), *simulated_shape),
            dtype="f"
        )
        spherex_dataset = hdf.create_dataset(
            "spherex_arrays",
            shape=(len(results), *spherex_shape),
            dtype="f"
        )

        # Write data into the datasets
        for i, result in enumerate(results):
            simulated_dataset[i] = result["simulated_array"]
            spherex_dataset[i] = result["spherex_array"]

    print(f"Results successfully saved to {output_path}")






def task_wrapper(args):
    """
    Wrapper function to pass multiple arguments to generate_and_process.
    """
    return generate_and_process(*args)


def main(num_images, probLensing=1, output_path="output"):
    """
    Main function to generate and process galaxy images in parallel.

    Parameters:
        num_images (int): Number of images to generate and process.
        probLensing (float): Probability of lensed object. 1 = All objects are lensed, 0 = none. Default is 100%.
        output_path (str): Path to save the results file.
    """
    # Prepare arguments for task_wrapper
    args = [(index, probLensing) for index in range(num_images)]

    # Create a pool of workers and process images in parallel
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap(task_wrapper, args),
                total=num_images,
                desc="Processing images"
            )
        )
    # Save the results
    save_results(results, output_path)
    return results

if __name__ == "__main__":
    num_images = 5  # Adjust this value as needed, prob needs to be 500+ lmao
    results = main(num_images, probLensing = 0.5, output_path='output/test3.h5')


'''