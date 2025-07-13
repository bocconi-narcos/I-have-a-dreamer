import os
import sys
import torch
import pickle
import h5py
import argparse
import numpy as np

def load_buffer(buffer_path):
    if buffer_path.endswith('.pkl'):
        with open(buffer_path, 'rb') as f:
            buffer = pickle.load(f)
    elif buffer_path.endswith('.pt'):
        buffer = torch.load(buffer_path, map_location='cpu')
    elif buffer_path.endswith('.hdf5') or buffer_path.endswith('.h5'):
        with h5py.File(buffer_path, 'r') as f:
            num_transitions = f['state'].shape[0]
            buffer = {k: np.array(f[k]) for k in f.keys()}
    else:
        raise ValueError(f"Unsupported buffer file format: {buffer_path}")
    return buffer

def dict_to_tensor_buffer(buffer_dict):
    # Convert all arrays to torch tensors
    tensor_buffer = {}
    for k, v in buffer_dict.items():
        if isinstance(v, np.ndarray):
            tensor_buffer[k] = torch.from_numpy(v)
        elif isinstance(v, list):
            # Try to convert to tensor if possible
            try:
                tensor_buffer[k] = torch.tensor(v)
            except Exception:
                tensor_buffer[k] = v
        elif torch.is_tensor(v):
            tensor_buffer[k] = v
        else:
            tensor_buffer[k] = v
    return tensor_buffer

def list_to_tensor_buffer(buffer_list):
    # Assume list of dicts, convert to dict of lists
    keys = buffer_list[0].keys()
    out = {k: [] for k in keys}
    for item in buffer_list:
        for k in keys:
            out[k].append(item[k])
    # Recursively convert
    return dict_to_tensor_buffer(out)

def preprocess_buffer(input_path, output_path):
    print(f"Loading buffer from {input_path}")
    buffer = load_buffer(input_path)
    if isinstance(buffer, dict) and 'state' in buffer:
        tensor_buffer = dict_to_tensor_buffer(buffer)
    elif isinstance(buffer, list):
        tensor_buffer = list_to_tensor_buffer(buffer)
    else:
        raise ValueError("Unknown buffer format!")
    print(f"Saving fast buffer to {output_path}")
    torch.save(tensor_buffer, output_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess replay buffer for fast loading.")
    parser.add_argument('input', type=str, help='Input buffer file (.pkl, .pt, .hdf5, .h5)')
    parser.add_argument('output', type=str, help='Output .pt file for fast loading')
    args = parser.parse_args()
    preprocess_buffer(args.input, args.output)

if __name__ == "__main__":
    main() 