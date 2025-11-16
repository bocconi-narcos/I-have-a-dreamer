import os
import json

def check_runs_status():
    """Check the status of wandb runs."""
    wandb_dir = "wandb"
    
    print("🔍 Checking wandb runs status...")
    print("=" * 50)
    
    synced_runs = []
    local_runs = []
    
    for item in os.listdir(wandb_dir):
        if item.startswith("run-"):
            run_path = os.path.join(wandb_dir, item)
            if os.path.isdir(run_path):
                history_file = os.path.join(run_path, "wandb-history.jsonl")
                
                if os.path.exists(history_file):
                    # Check if run is synced
                    try:
                        with open(history_file, 'r') as f:
                            lines = f.readlines()
                            last_line = lines[-1] if lines else ""
                            if '"synced": true' in last_line:
                                synced_runs.append(item)
                            else:
                                local_runs.append(item)
                    except Exception:
                        local_runs.append(item)
                else:
                    local_runs.append(item)
    
    print(f"✅ Synced runs: {len(synced_runs)}")
    print(f"⏳ Local-only runs: {len(local_runs)}")
    
    if synced_runs:
        print(f"\n✅ Recent synced runs:")
        for run in synced_runs[-5:]:  # Show last 5
            print(f"   {run}")
    
    if local_runs:
        print(f"\n📊 Local-only runs:")
        for run in local_runs[-5:]:  # Show last 5
            print(f"   {run}")
    
    print(f"\n🌐 Check your runs at: https://wandb.ai/[your-username]/[your-project]")

if __name__ == "__main__":
    check_runs_status() 