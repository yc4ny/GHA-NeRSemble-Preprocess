import argparse
import os
import subprocess


def main(id, gpu, start_stage):
    start_stage = int(start_stage)
    print(f"Processing ID: {id} on GPU: {gpu}")

    # Step 1: Preprocess the data
    if start_stage <= 1:
        print("Starting preprocessing...")
        subprocess.run(f"python preprocess/preprocess_nersemble.py --id {id}", shell=True)

    # Step 2: Sample the data
    if start_stage <= 2:
        print("Sampling data...")
        subprocess.run(f"python sample_data.py --input_dir preprocessed_nersemble/{id} --output_base preprocessed_nersemble_part", shell=True)

    # Step 3 Perform background removal
    if start_stage <= 3:
        print("Removing background...")
        os.chdir("BackgroundMattingV2")
        subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python remove_background_nersemble.py --id {id}", shell=True)
        os.chdir("..")

    # Step 4: Change to Multiview-3DMM-Fitting directory and prepare configuration
    if start_stage <= 4:
        os.chdir("Multiview-3DMM-Fitting")
        config_content = f"""
    image_folder: '../preprocessed_nersemble_part/{id}/images'
    camera_folder: '../preprocessed_nersemble_part/{id}/cameras'
    landmark_folder: '../preprocessed_nersemble_part/{id}/landmarks'
    param_folder: '../preprocessed_nersemble_part/{id}/params'
    gpu_id: 0
    camera_ids: ['220700191', '221501007', '222200036', '222200037', '222200038', '222200039', '222200040', '222200041',
                '222200042', '222200043', '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']
    image_size: 2048
    face_model: 'BFM'
    reg_id_weight: 1e-7
    reg_exp_weight: 1e-7
    save_vertices: True
    visualize: False
    """
        with open(f"config/NeRSemble_{id}.yaml", "w") as file:
            file.write(config_content)

        print("Detecting landmarks...")
        subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python detect_landmarks.py --config config/NeRSemble_{id}.yaml", shell=True)

        print("Performing fitting...")
        subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python fitting.py --config config/NeRSemble_{id}.yaml", shell=True)

        os.chdir("..")

    # Step 5: Training Stage 1 
    if start_stage <= 5:
        mesh_config = f"""
gpu_id: 0
load_meshhead_checkpoint: 'checkpoints/meshhead_NeRSemble{id}/meshhead_latest'
lr_net: 1e-3
lr_lmk: 1e-4
batch_size: 1

dataset:
  dataroot: 'preprocessed_nersemble_part/{id}'
  camera_ids: ['220700191', '221501007', '222200036', '222200037', '222200038', '222200039', '222200040', '222200041',
               '222200042', '222200043', '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']
  original_resolution: 2048
  resolution: 256
  num_sample_view: 4

meshheadmodule:
  geo_mlp: [27, 256, 256, 256, 256, 256, 132]
  exp_color_mlp: [192, 256, 256, 32]
  pose_color_mlp: [182, 128, 32]
  exp_deform_mlp: [91, 256, 256, 256, 256, 256, 3]
  pose_deform_mlp: [81, 256, 256, 3]
  pos_freq: 4
  model_bbox: [[-1.6, 1.6], [-1.7, 1.8], [-2.5, 1.0]]
  dist_threshold_near: 0.1
  dist_threshold_far: 0.25
  deform_scale: 0.3
  subdivide: False

recorder:
  name: 'meshhead_NeRSemble{id}'
  logdir: 'log/meshhead_NeRSemble{id}'
  checkpoint_path: 'checkpoints'
  result_path: 'results'
  save_freq: 1000
  show_freq: 100
"""
        with open(f"config/train_meshhead_N{id}.yaml", "w") as file:
            file.write(mesh_config)

        print("Training geometry guidance model...")
        subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python train_meshhead.py --config config/train_meshhead_N{id}.yaml", shell=True)

    # Step 6: Training Stage 2
    if start_stage <= 6:
        gaussian_config = f"""
gpu_id: 0
load_meshhead_checkpoint: 'checkpoints/meshhead_NeRSemble{id}/meshhead_latest'
load_gaussianhead_checkpoint: 'checkpoints/gaussianhead_NeRSemble{id}/gaussianhead_latest'
load_supres_checkpoint: 'checkpoints/gaussianhead_NeRSemble{id}/supres_latest'
load_delta_poses_checkpoint: 'checkpoints/gaussianhead_NeRSemble{id}/delta_poses_latest'
lr_net: 1e-4
lr_pose: 1e-5
batch_size: 1
optimize_pose: True

dataset:
  dataroot: 'preprocessed_nersemble_part/{id}'
  camera_ids: ['220700191', '221501007', '222200036', '222200037', '222200038', '222200039', '222200040', '222200041',
               '222200042', '222200043', '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']
  original_resolution: 2048
  resolution: 2048

meshheadmodule:
  geo_mlp: [27, 256, 256, 256, 256, 256, 132]
  exp_color_mlp: [192, 256, 256, 32]
  pose_color_mlp: [182, 128, 32]
  exp_deform_mlp: [91, 256, 256, 256, 256, 256, 3]
  pose_deform_mlp: [81, 256, 256, 3]
  pos_freq: 4
  model_bbox: [[-1.6, 1.6], [-1.7, 1.8], [-2.5, 1.0]]
  dist_threshold_near: 0.1
  dist_threshold_far: 0.25
  deform_scale: 0.3
  subdivide: False

supresmodule:
  input_dim: 32
  output_dim: 3
  network_capacity: 32

gaussianheadmodule:
  num_add_mouth_points: 3000
  exp_color_mlp: [192, 256, 256, 32]
  pose_color_mlp: [182, 128, 32]
  exp_deform_mlp: [91, 256, 256, 256, 256, 256, 3]
  pose_deform_mlp: [81, 256, 256, 3]
  exp_attributes_mlp: [192, 256, 256, 256, 8]
  pose_attributes_mlp: [182, 128, 128, 8]
  exp_coeffs_dim: 64
  pos_freq: 4
  dist_threshold_near: 0.1
  dist_threshold_far: 0.25
  deform_scale: 0.3
  attributes_scale: 0.2

recorder:
  name: 'gaussianhead_NeRSemble{id}'
  logdir: 'log/gaussianhead_NeRSemble{id}'
  checkpoint_path: 'checkpoints'
  result_path: 'results'
  save_freq: 1000
  show_freq: 100
"""
        with open(f"config/train_gaussianhead_N{id}.yaml", "w") as file:
            file.write(gaussian_config)

        print("Training Gaussian head model...")
        subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python train_gaussianhead.py --config config/train_gaussianhead_N{id}.yaml", shell=True)

    print("All training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models for given dataset ID and start from a specified stage.')
    parser.add_argument('--id', required=True, type=str, help='Dataset ID to process')
    parser.add_argument('--gpu', required=True, type=int, help='GPU number to use')
    parser.add_argument('--start_stage', required=False, type=str, help='Starting stage (1-6)', default = '1')
    args = parser.parse_args()

    main(args.id, args.gpu, args.start_stage)
