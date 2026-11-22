import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source image folder')
    parser.add_argument('--dst', type=str, required=True, help='Destination sparse image folder')
    parser.add_argument('--step', type=int, default=10, help='Sampling step (e.g., 10 for 1/10)')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted([f for f in os.listdir(args.src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    kept = files[::args.step]
    for i, f in enumerate(kept):
        ext = os.path.splitext(f)[1]
        shutil.copy(os.path.join(args.src, f), os.path.join(args.dst, f"{i:06d}{ext}"))
    
    print(f"Kept {len(kept)} / {len(files)} frames, step={args.step}")

if __name__ == '__main__':
    main()
