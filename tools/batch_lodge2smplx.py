import os
import sys
import glob
import subprocess

def main(src_dir, out_dir, fps=30):
    os.makedirs(out_dir, exist_ok=True)

    # 先扫到所有名字像 .npy 的路径
    all_paths = sorted(glob.glob(os.path.join(src_dir, "*.npy")))

    # 只保留“真文件”，跳过目录
    files = [p for p in all_paths if os.path.isfile(p)]

    print(f"found {len(all_paths)} paths ending with .npy")
    print(f"valid npy files: {len(files)}")

    skipped = [p for p in all_paths if not os.path.isfile(p)]
    if skipped:
        print("skipped non-file paths:")
        for p in skipped[:20]:
            print("  ", p)

    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        out_pkl = os.path.join(out_dir, name + "_smplx.pkl")
        cmd = [
            sys.executable,
            "-m",
            "tools.lodge2smplx_pkl",
            f,
            out_pkl,
            str(fps),
        ]
        print(f"[{i+1}/{len(files)}] converting: {name}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("FAILED:", f)
            break

    print("done.")

if __name__ == "__main__":
    src_dir = sys.argv[1]
    out_dir = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    main(src_dir, out_dir, fps)