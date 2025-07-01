from replay_buffer_dataset import ReplayBufferDataset

# Set these variables to match your buffer and config
buffer_path = "/Users/vittoriogaravelli/GitHub/GitHub/I-have-a-dreamer/data/buffer.h5"  # or .hdf5
num_color_selection_fns = 5
num_selection_fns = 5
num_transform_actions = 5
num_arc_colors = 10
state_shape = (3, 32, 32)  # Example shape, adjust as needed
mode = "color_only"  # or "selection_color", "end_to_end"
num_samples = 10  # Number of samples to load for inspection

# Load the dataset
dataset = ReplayBufferDataset(
    buffer_path=buffer_path,
    num_color_selection_fns=num_color_selection_fns,
    num_selection_fns=num_selection_fns,
    num_transform_actions=num_transform_actions,
    num_arc_colors=num_arc_colors,
    state_shape=state_shape,
    mode=mode,
    num_samples=num_samples
)

# Print a few samples
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}:")
    for key, value in sample.items():
        if key == "state":
            print(f"  {key}:")
            print(value)  # or print(value[:5]) for just the first 5 rows
        else:
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    print("-" * 40)
