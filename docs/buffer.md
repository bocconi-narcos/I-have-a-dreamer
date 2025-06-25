# Buffer/Data Structure

The training data is a **buffer**: a list of dictionaries, each representing a transition. Each entry contains:

- `state`: (H, W) — The current grid state.
- `target_state`: (H, W) — The target grid state.
- `color_in_state`: int — Number of colors in the original grid.
- `action`: dict — Contains:
  - `'colour'`: int — Color selection index.
  - `'selection'`: int — Selection index (used for selection mask prediction).
  - `'transform'`: int — Transform index.
- `colour`: int — The color resulting from the color selection.
- `selection_mask`: (H, W) — The selection mask resulting from the selection (used as ground truth for selection mask prediction).
- `reward`: float — Reward for the transition.
- `next_state`: (H, W) — The next grid state.
- `done`: bool — Whether the episode is done.
- `info`: dict (optional) — For meta data purposes.

If no buffer file is found, the code generates dummy data with this structure. 