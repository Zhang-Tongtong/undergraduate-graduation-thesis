import sys
import pickle

with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())
for k, v in data.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, type(v), v)