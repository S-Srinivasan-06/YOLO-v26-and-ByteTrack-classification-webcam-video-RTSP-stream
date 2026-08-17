import csv
import math
import sys
from collections import deque
from pathlib import Path

# Setup paths based on project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "outputs" / "log.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "outputs" / "metric.csv"

def safe_float(val):
    """Safely cast string to float, returning None if invalid."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Input file {INPUT_CSV} not found.")
        sys.exit(1)

    # Primary target columns
    targets = ['count', 'cov', 'wcount', 'speed', 'dirx', 'diry', 'consist']
    windows = [5, 10]
    
    # Read the entire input CSV
    rows = []
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Error: Input file is empty.")
            sys.exit(1)
            
        # Verify target columns exist
        col_indices = {}
        for t in targets:
            try:
                col_indices[t] = header.index(t)
            except ValueError:
                print(f"Warning: Column '{t}' not found in input CSV. It will be skipped.")
        
        # We only process targets that are actually in the CSV header
        actual_targets = [t for t in targets if t in col_indices]
        
        for row in reader:
            rows.append(row)
            
    # Prepare new header
    new_header = list(header)
    for w in windows:
        for t in actual_targets:
            new_header.extend([f"{t}_sma{w}", f"{t}_res{w}", f"{t}_std{w}", f"{t}_z{w}"])
            
    # Process rows and calculate metrics
    processed_rows = []
    rolling_data = {w: {t: deque(maxlen=w) for t in actual_targets} for w in windows}
    
    for row in rows:
        new_row = list(row)
        current_vals = {}
        
        # Extract current values and update rolling windows
        for t in actual_targets:
            # Handle potential short rows missing some columns
            if col_indices[t] < len(row):
                val = safe_float(row[col_indices[t]])
            else:
                val = None
                
            current_vals[t] = val
            
            for w in windows:
                if val is not None:
                    rolling_data[w][t].append(val)
                else:
                    # Clear window if we encounter missing data to prevent stale calculations
                    rolling_data[w][t].clear()
                    
        # Calculate features for each window and target
        for w in windows:
            for t in actual_targets:
                window_vals = rolling_data[w][t]
                
                # Check if window is completely filled
                if len(window_vals) == w:
                    sma = sum(window_vals) / w
                    current_val = current_vals[t]
                    residual = current_val - sma
                    
                    # Sample standard deviation (Bessel's correction, ddof=1)
                    if w > 1:
                        variance = sum((x - sma) ** 2 for x in window_vals) / (w - 1)
                        std = math.sqrt(variance)
                    else:
                        std = 0.0
                        
                    # Z-score with epsilon to avoid division by zero
                    z_score = residual / (std + 1e-6)
                    
                    new_row.extend([f"{sma:.4f}", f"{residual:.4f}", f"{std:.4f}", f"{z_score:.4f}"])
                else:
                    # Pad early rows where window size hasn't been met yet
                    new_row.extend(['', '', '', ''])
                    
        processed_rows.append(new_row)
        
    # Write output to metric.csv
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(processed_rows)
        
    print(f"Successfully processed {len(processed_rows)} rows.")
    print(f"Metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
