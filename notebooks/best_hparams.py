import os
import shutil
import numpy as np
from collections import OrderedDict

def safe_eval(text):
    # Replace `nan` safely
    text = text.replace('nan', 'float("nan")')

    # Now allow OrderedDict inside eval
    safe_globals = {
        "OrderedDict": OrderedDict,
        "float": float
    }
    try:
        data = eval(text, {"__builtins__": None}, safe_globals)
        return data
    except Exception as e:
        raise ValueError(f"safe_eval failed: {e}")

def extract_losses(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    try:
        data = safe_eval(content)
        nll_loss = data.get('nll_loss', (float('inf'),))[0]
        l1_loss = data.get('l1_loss', (float('inf'),))[0]
        total_loss = data.get('total_loss', (float('inf'),))[0]
        return nll_loss, l1_loss, total_loss
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return float('inf'), float('inf'), float('inf')

def main(input_folder):
    best_folder = os.path.join(input_folder, 'best')
    os.makedirs(best_folder, exist_ok=True)

    file_losses = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            full_path = os.path.join(input_folder, filename)
            nll_loss, l1_loss, total_loss = extract_losses(full_path)
            file_losses.append((full_path, nll_loss, l1_loss, total_loss))
    
    lowest_nll = sorted(file_losses, key=lambda x: x[1])[:3]
    lowest_l1 = sorted(file_losses, key=lambda x: x[2])[:3]
    lowest_total = sorted(file_losses, key=lambda x: x[3])[:3]
    
    selected_files = set()
    for fileset in [lowest_nll, lowest_l1, lowest_total]:
        for filepath, _, _, _ in fileset:
            selected_files.add(filepath)

    for filepath in selected_files:
        shutil.copy(filepath, best_folder)

    # ==== Print report ====
    print("\n=== Summary Report ===")
    def print_file_list(title, fileset):
        print(f"\n{title}:")
        for fullpath, loss1, loss2, loss3 in fileset:
            filename = os.path.basename(fullpath)
            if title == "Lowest NLL Loss":
                print(f"  {filename} | nll_loss = {loss1:.4f}")
            elif title == "Lowest L1 Loss":
                print(f"  {filename} | l1_loss = {loss2:.4f}")
            elif title == "Lowest Total Loss":
                print(f"  {filename} | total_loss = {loss3:.4f}")

    print_file_list("Lowest NLL Loss", lowest_nll)
    print_file_list("Lowest L1 Loss", lowest_l1)
    print_file_list("Lowest Total Loss", lowest_total)

    print(f"\nCopied {len(selected_files)} files into {best_folder}")

if __name__ == "__main__":
    input_folder = "results/ROME_MODIFIED/summaries/new"  # Change if needed
    main(input_folder)
