# Build hi-res MP4s from the Visible Human PNG slices.
# One video per body section, plus a whole-body (head-to-toe) video, for Male and Female.
# Usage: python3 make_videos.py [fps] [crf]
# Empty files and solid-black placeholder slices (<100 KB) are skipped.
import os, sys, subprocess, tempfile
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Videos")
FPS = sys.argv[1] if len(sys.argv) > 1 else "30"
CRF = sys.argv[2] if len(sys.argv) > 2 else "16"
ORDER = ["head", "thorax", "abdomen", "pelvis", "thighs", "legs"]
MIN_SIZE = 100_000

def slices(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith(".png") and not f.startswith(".") and os.path.getsize(os.path.join(folder, f)) >= MIN_SIZE)

def encode(paths, dest):
    if os.path.exists(dest):
        print(f"skip (exists) {dest}", flush=True); return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(paths):
            os.symlink(p, os.path.join(tmp, f"{i:05d}.png"))
        print(f"encoding {len(paths)} frames -> {dest}", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-framerate", FPS, "-i", os.path.join(tmp, "%05d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", dest + ".part.mp4"], check=True)
        os.replace(dest + ".part.mp4", dest)

for sex in ["Male", "Female"]:
    src = os.path.join(ROOT, f"{sex}-Images", "PNG_format")
    whole = {}
    for i, sec in enumerate(ORDER, 1):
        paths = slices(os.path.join(src, sec))
        for p in paths: whole.setdefault(os.path.basename(p), p)  # sections overlap; keep each slice once
        encode(paths, os.path.join(OUT, sex, f"{sex}_{i}_{sec}.mp4"))
    encode([whole[k] for k in sorted(whole)], os.path.join(OUT, sex, f"{sex}_0_whole_body.mp4"))
print("DONE", flush=True)
