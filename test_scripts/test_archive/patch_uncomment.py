"""Uncomment the run_solvator_cluster and inspect_job block in server_with_product.py."""

with open("server_with_product.py", "r") as f:
    lines = f.readlines()

# Find the two ''' lines that bracket the commented block
triple_indices = [i for i, ln in enumerate(lines) if ln.strip() == "'''"]
if len(triple_indices) < 2:
    print(f"ERROR: expected at least 2 triple-quote lines, found {len(triple_indices)}")
else:
    # Remove both ''' lines (remove later one first to preserve indices)
    # Also remove the "from typing import ..." line right after the closing '''
    close_idx = triple_indices[-1]
    open_idx = triple_indices[-2]

    # Check if line after closing ''' is a redundant import
    next_line = lines[close_idx + 1].strip() if close_idx + 1 < len(lines) else ""

    # Remove closing '''
    del lines[close_idx]
    # If next line was redundant import, remove it too
    if next_line.startswith("from typing import"):
        del lines[close_idx]  # same index since we deleted the ''' already

    # Remove opening '''
    del lines[open_idx]

    with open("server_with_product.py", "w") as f:
        f.writelines(lines)
    print("OK: uncommented run_solvator_cluster and inspect_job")
