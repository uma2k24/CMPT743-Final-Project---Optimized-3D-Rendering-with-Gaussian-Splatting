# Hybrid Mesh + Gaussian Splatting Baseline

It is intentionally scoped as a project baseline rather than a full reproduction of Fantasia3D or the original 3D Gaussian Splatting paper.

## What it implements

The code follows the pipeline proposed in the documents:

1. coarse mesh prior
2. surface sampling on the mesh
3. Gaussian splat initialization around sampled surface points
4. anchored optimization with tether regularization
5. appearance refinement under a lightweight prompt-conditioned prior
6. multi-view rendering from the learned Gaussian cloud

The implementation uses:

- a simple OBJ loader or built-in primitive meshes
- sampled surface anchors and normals
- a learnable Gaussian cloud with position, scale, opacity, and color
- a differentiable splat renderer in PyTorch
- a hybrid loss:

```text
L = L_image + lambda_tether * L_tether + lambda_appearance * L_appearance
    + lambda_scale * L_scale + lambda_opacity * L_opacity
```

`L_appearance` is a lightweight prompt-palette prior that stands in for diffusion guidance. It is the correct place to later integrate SDS / diffusion-based appearance supervision.

## Quick start

Install dependencies in your environment:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the default demo:

```powershell
& .\.venv\Scripts\python.exe .\main.py --steps 200 --out-dir .\outputs\demo
```

This will:

- create a coarse primitive mesh from the prompt
- sample mesh surface points
- initialize anchored Gaussian splats
- build proxy multi-view targets
- optimize the splats
- save rendered views and target comparisons to `outputs/demo`

## Using a real mesh

If you export a coarse mesh from Fantasia3D or another text-to-3D system as OBJ, plug it in like this:

```powershell
& .\.venv\Scripts\python.exe .\main.py --mesh path\to\coarse_mesh.obj --prompt "stone statue" --steps 300
```

## Key limitations

- The renderer uses isotropic screen-space splats, not the full anisotropic covariance formulation from 3DGS.
- The appearance prior is a placeholder for diffusion guidance.
- Proxy target images are generated procedurally so the baseline is self-contained.

## Recommended next extensions

1. Replace the prompt-palette loss with diffusion/SDS guidance.
2. Swap the proxy targets for real multi-view images or rendered observations.
3. Upgrade the renderer to anisotropic covariance splats and tile-based compositing.
4. Add mesh-aware editing constraints so the mesh stays the explicit structural representation.
