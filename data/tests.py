import h5py

with h5py.File("buffer.h5", "r") as f:
    print("Top-level keys:", list(f.keys()))
