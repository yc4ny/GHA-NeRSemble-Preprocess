import argparse
import os
import shutil
from tqdm import tqdm

def copy_files(src, dst, pattern=None):
    """
    Copy files from src to dst. If pattern is None, copy all files.
    If pattern is not None, copy only files that match the pattern.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    files = os.listdir(src)
    for file in files:
        if pattern is None or file.endswith(pattern):
            src_file_path = os.path.join(src, file)
            dst_file_path = os.path.join(dst, file)
            if os.path.isfile(src_file_path):
                shutil.copy(src_file_path, dst_file_path)

def copy_directories(src, dst, step):
    """
    Copy every 'step'th directory from src to dst.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    directories = sorted(os.listdir(src))
    for i, directory in tqdm(enumerate(directories)):
        if i % step == 0:
            src_dir_path = os.path.join(src, directory)
            dst_dir_path = os.path.join(dst, directory)
            if os.path.isdir(src_dir_path):
                shutil.copytree(src_dir_path, dst_dir_path)

def main(input_dir, output_base):
    # Determine the final output directory based on input_dir base name
    base_name = os.path.basename(input_dir)
    final_output_dir = os.path.join(output_base, base_name)

    # Paths to the specific folders
    background_src = os.path.join(input_dir, 'background')
    cameras_src = os.path.join(input_dir, 'cameras')
    images_src = os.path.join(input_dir, 'images')
    
    background_dst = os.path.join(final_output_dir, 'background')
    cameras_dst = os.path.join(final_output_dir, 'cameras')
    images_dst = os.path.join(final_output_dir, 'images')
    
    # Copy all jpg files from background
    copy_files(background_src, background_dst, '.jpg')
    
    # Copy every 10th folder from cameras and images
    copy_directories(cameras_src, cameras_dst, 10)
    copy_directories(images_src, images_dst, 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy specific folders and files.')
    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--output_base', type=str, help='Base output directory path where input dir structure will be mirrored')
    args = parser.parse_args()

    main(args.input_dir, args.output_base)
    print("FINISHED SAMPLING DATA!")
