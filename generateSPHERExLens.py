import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import autolens as al

import sys
from pathlib import Path

base_dir = Path.cwd()

autolens_config_path = base_dir / "autolens_config"
spherex_path = base_dir / "SPHEREx"

sys.path.append(str(autolens_config_path))
sys.path.append(str(spherex_path))


import SPHERExScripts as SPHEREx  # from SPHEREX
import gcluster10 as galaxy #galaxy generating function



def generate_and_process(index):
    """
    Generate a lensed galaxy cluster image, process it to SPHEREx resolution, and return the results.

    Parameters:
        index (int): Index of the process for tracking.

    Returns:
        dict: A dictionary containing the raw and processed images and their numpy arrays.
    """
    # Generate the raw lensed galaxy cluster image
    raw = (galaxy.wrapperFunction(verbose=0))
    
    # Process the image to SPHEREx resolution
    convolved, binned = SPHEREx.processImg(raw)
    
    # Return data in dictionary format
    return {
        "index": index,
        "raw_image": raw,
        "spherex_image": binned,
        "raw_array": raw,
        "spherex_array": binned
    }

import numpy as np
import matplotlib.pyplot as plt
import os

def save_results(results, array_path="results_arrays.npz", image_dir="images"):
    """
    Save results including numpy arrays and images.

    Parameters:
        results (list): List of result dictionaries from generate_and_process.
        array_path (str): Path to save the numpy arrays (.npz file).
        image_dir (str): Directory to save the images.
    """
    # Ensure the image directory exists
    os.makedirs(image_dir, exist_ok=True)

    # Save the numpy arrays to an .npz file
    np_arrays = {
        f"raw_array_{res['index']}": res["raw_array"] for res in results
    }
    np_arrays.update({
        f"spherex_array_{res['index']}": res["spherex_array"] for res in results
    })
    np.savez(array_path, **np_arrays)
    print(f"Numpy arrays saved to {array_path}")

    # Save the actual images
    for res in results:
        index = res["index"]

        # Save Raw Image
        raw_image_path = os.path.join(image_dir, f"raw_image_{index}.png")
        plt.imsave(raw_image_path, res["raw_array"], cmap="inferno")
        
        # Save SPHEREx Resolution Image
        spherex_image_path = os.path.join(image_dir, f"spherex_image_{index}.png")
        plt.imsave(spherex_image_path, res["spherex_array"], cmap="inferno")
        
        print(f"Images saved for index {index}:")
        print(f"  Raw Image: {raw_image_path}")
        print(f"  SPHEREx Image: {spherex_image_path}")


def main(num_images, output_path="results_table.csv"):
    """
    Main function to generate and process galaxy images in parallel.

    Parameters:
        num_images (int): Number of images to generate and process.
        output_path (str): Path to save the results file.
    """
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap(generate_and_process, range(num_images)),
                total=num_images,
                desc="Processing images"
            )
        )
    save_results(results, output_path)

if __name__ == "__main__":
    num_images = 5  # Adjust this value as needed, prob needs to be 500+ lmao
    main(num_images)


