# inspect_gmr_pkl.py
import sys, pickle

with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, dict):
    print("keys:", list(data.keys()))
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v))