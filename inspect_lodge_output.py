# inspect_lodge_output.py
import os
import sys
import json
import numpy as np

def inspect_file(path):
    print("=" * 80)
    print("FILE:", path)

    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        print("type:", type(arr))
        if isinstance(arr, np.ndarray):
            print("shape:", arr.shape, "dtype:", arr.dtype)
            if arr.dtype == object:
                try:
                    obj = arr.item()
                    print("object keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
                except Exception as e:
                    print("cannot item():", e)

    elif path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        print("keys:", data.files)
        for k in data.files:
            v = data[k]
            print(f"  {k}: shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print("type:", type(obj))
        if isinstance(obj, dict):
            print("keys:", list(obj.keys()))
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: type={type(v)}")

def main(root):
    exts = (".npy", ".npz", ".pkl", ".bvh", ".json", ".mp4")
    all_files = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.endswith(exts):
                all_files.append(os.path.join(dp, f))
    all_files.sort()

    print("found files:", len(all_files))
    for p in all_files[:50]:
        print(p)

    print("\n\nTrying to inspect npy/npz/pkl files...\n")
    for p in all_files:
        if p.endswith((".npy", ".npz", ".pkl")):
            try:
                inspect_file(p)
            except Exception as e:
                print("ERROR:", p, e)

if __name__ == "__main__":
    root = sys.argv[1]
    main(root)