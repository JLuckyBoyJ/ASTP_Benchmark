import json
import glob
import os

cache_dir = "data/cache/moh/synthetic/atsp_gls"
json_files = glob.glob(os.path.join(cache_dir, "seed_pop_*.json"))

for file_path in json_files:
    print(f"Cleaning {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned_count = 0
    for item in data:
        code = item.get("best_sol", "")
        if "local_opt_tour[i] - 1" in code or "local_opt_tour[(i + 1) % n] - 1" in code:
            code = code.replace("local_opt_tour[i] - 1", "local_opt_tour[i]")
            code = code.replace("local_opt_tour[(i + 1) % n] - 1", "local_opt_tour[(i + 1) % n]")
            code = code.replace("(local_opt_tour[i] - 1)", "local_opt_tour[i]")
            code = code.replace("(local_opt_tour[(i + 1) % n] - 1)", "local_opt_tour[(i + 1) % n]")
            code = code.replace("local_opt_tour[k] - 1", "local_opt_tour[k]")
            code = code.replace("local_opt_tour[(k + 1) % n] - 1", "local_opt_tour[(k + 1) % n]")
            item["best_sol"] = code
            cleaned_count += 1
            
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"  Fixed {cleaned_count} items in {file_path}")

print("All seed cache files cleaned successfully!")
