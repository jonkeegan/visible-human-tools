# Build a grey surface mesh (skin) of a Visible Human body from the crown down to just below the shoulders.
# Pixels are 0.33 mm; female slices are 0.33 mm apart (isotropic), male slices 1 mm apart.
# Writes Models/<sex>_shoulders_up_full.stl (full res), a decimated copy, and a shaded front preview PNG.
# Run with the project venv: .venv/bin/python make_head_model.py Female|Male [last_level] [target_tris]
import os, sys, numpy as np
from PIL import Image
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
from skimage.measure import marching_cubes
import trimesh, fast_simplification
from make_explorer_assets import all_slices

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Models")
SEX = sys.argv[1] if len(sys.argv) > 1 else "Female"
# last slice level: just below the armpits; body region (x0, y0, x1, y1) in the axial slices;
# slice spacing in pixels; smoothing sigma in voxels (slice, row, col) - more along z for the male's 1 mm steps
CFG = {"Female": dict(last=1340, crop=(250, 90, 1860, 960), dz=1.0, sigma=(1.0, 1.0, 1.0)),
       "Male":   dict(last=1400, crop=(150, 90, 1860, 1040), dz=3.0, sigma=(1.0, 1.5, 1.5))}[SEX]
LAST = int(sys.argv[2]) if len(sys.argv) > 2 else CFG["last"]
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000  # triangles in the decimated copy
X0, Y0, X1, Y1 = CFG["crop"]
MM, CHUNK, MARGIN = 0.33, 96, 6
NAME = f"{SEX.lower()}_shoulders_up"

def mask(path):
    a = np.asarray(Image.open(path).convert("RGB").crop((X0, Y0, X1, Y1))).astype(np.int16)
    tissue = (a[..., 0] - a[..., 2] > 18) & (a[..., 0] > 45)  # gel is blue, tissue red/pink/yellow
    tissue = ndimage.binary_closing(ndimage.binary_opening(tissue, iterations=2), iterations=4)
    tissue = ndimage.binary_fill_holes(tissue)  # solid inside: we only want the skin surface
    lab, n = ndimage.label(tissue)
    if n:
        sizes = ndimage.sum(tissue, lab, range(1, n + 1))
        boxes = ndimage.find_objects(lab)
        keep = [k + 1 for k, sz in enumerate(sizes) if sz >= max(4000, sizes.max() * 0.02)
                and sz / np.prod([s.stop - s.start for s in boxes[k]]) < 0.9]  # drop specks and sticky-note labels
        tissue = np.isin(lab, keep)
    return np.packbits(tissue, axis=1)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    slices = [p for f, p in all_slices(SEX) if int(f[-8:-4] if SEX == "Male" else f[3:7]) <= LAST]
    valid = [p if os.path.getsize(p) >= 100_000 else None for p in slices]
    for i, p in enumerate(valid):  # stand in the previous good slice for placeholders
        if p is None: valid[i] = valid[i - 1]
    print(f"reading {len(valid)} slices", flush=True)
    W = X1 - X0
    with ProcessPoolExecutor(6) as ex:
        vol = np.stack([np.unpackbits(m, axis=1)[:, :W].astype(bool) for m in ex.map(mask, valid, chunksize=4)])
    vol = np.pad(vol, 2)  # closed surface at the crop edges and the bottom cut
    print("volume", vol.shape, flush=True)

    # smooth and mesh in overlapping z-chunks so memory stays bounded; shared boundary planes give identical vertices
    verts, faces, nv = [], [], 0
    Z = vol.shape[0]
    for z0 in range(0, Z - 1, CHUNK):
        z1 = min(Z, z0 + CHUNK + 1)
        a, b = max(0, z0 - MARGIN), min(Z, z1 + MARGIN)
        sm = ndimage.gaussian_filter(vol[a:b].astype(np.float32), CFG["sigma"])[z0 - a:z1 - a]
        v, f, _, _ = marching_cubes(sm, 0.5, spacing=(CFG["dz"], 1, 1), step_size=1, allow_degenerate=False)
        v[:, 0] += z0 * CFG["dz"]
        verts.append(v); faces.append(f + nv); nv += len(v)
        print(f"  meshed slices {z0}-{z1}: {len(f):,} triangles", flush=True)
    v = np.concatenate(verts); f = np.concatenate(faces)
    # volume axes (slice, row, col) -> X = col (patient's left at +X), Y = -row (face toward -Y), Z = -slice (crown at top)
    pts = np.column_stack([v[:, 2], -v[:, 1], -v[:, 0]]) * MM
    mesh = trimesh.Trimesh(pts, f, process=False)
    mesh.merge_vertices()
    mesh.apply_translation(-mesh.bounds.mean(axis=0) * [1, 1, 0] - [0, 0, mesh.bounds[1, 2]])  # crown at z=0, centred
    if mesh.volume < 0: mesh.invert()
    print(f"full mesh: {len(mesh.faces):,} triangles, {len(mesh.vertices):,} vertices, "
          f"size {np.round(mesh.extents).astype(int)} mm, watertight={mesh.is_watertight}", flush=True)
    mesh.export(os.path.join(OUT, f"{NAME}_full.stl"))

    vs, fs = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1 - TARGET / len(mesh.faces))
    small = trimesh.Trimesh(vs, fs)
    small.export(os.path.join(OUT, f"{NAME}_{len(fs) // 1_000_000 or 1}M.stl"))
    print(f"decimated mesh: {len(fs):,} triangles", flush=True)

    # shaded front view straight from the volume: first tissue voxel seen from the front, lit from the upper left
    depth = np.where(vol.any(axis=1), vol.shape[1] - np.argmax(vol[:, ::-1, :], axis=1), 0).astype(np.float32)
    depth = ndimage.gaussian_filter(depth, 1.5)
    gz, gx = np.gradient(depth)
    n = np.dstack([-gx, -gz, np.ones_like(depth) * 1.5]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.4, -0.5, 0.77]); light /= np.linalg.norm(light)
    shade = np.clip((n @ light) * 0.85 + 0.15, 0, 1)
    img = np.where(vol.any(axis=1), shade * 210 + 25, 245).astype(np.uint8)
    im = Image.fromarray(img)
    im.resize((im.width, round(im.height * CFG["dz"])), Image.LANCZOS).save(os.path.join(OUT, f"{NAME}_preview.png"))
    print("DONE", flush=True)
