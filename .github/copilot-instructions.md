
# Riverborn — Copilot Coding Agent Instructions

> **Repository:** riverborn  
> **Owner:** lordmauve  
> **Audience:** GitHub Copilot coding agent

---

## 🏞️ Project Summary

Riverborn is a small Python 3.10+ game (PyWeek 39) that renders a canoeing
scene with ModernGL. It features a simple scene graph, shadow mapping, water
simulation, and editor tools. The game can run interactively (GLFW window) or
headless for tests (moderngl standalone context).

All code is under `src/riverborn`. Shaders, models, textures, and sounds are
package data, accessed via `importlib.resources`. Tests exercise shader
preprocessing, terrain generation, drawing, and shadow mapping using a headless
context with reference screenshots.

---

## 📦 Repo Stats & Tech

- **Size:** Small-to-medium Python project
- **Key dirs:** `src/riverborn` (~20 modules), `tests` (2 test files + reference
  PNGs), assets under `src/riverborn/{shaders,models,textures,sounds}`
- **Languages/tools:** Python 3.10–3.13, ModernGL, moderngl-window, numpy,
  pygame/pyglet, imageio, pyglm
- **Build backend:** flit
- **Package manager:** uv (preferred), or pip

---

## 🚀 Getting Started (Bootstrap)

**Always follow these steps in order:**

1. Ensure `uv` is installed (recommended) or use an existing venv. Python 3.13
   works (tested); project requires >=3.10.
2. From repo root, install dependencies with uv (preferred):
    - `uv sync --python 3.13`
    - _Tip: This uses `pyproject.toml` and `uv.lock`. It is fast and reproducible._
3. **Alternative** (not preferred): create and activate a venv, then
   `pip install -r requirements.txt`. Note: the repo’s bundled venv may lack pip;
   use `python -m ensurepip -U` then `python -m pip install -r requirements.txt`.
4. Verify imports headlessly:
    - `uv run -q python -c "import moderngl, numpy, imageio, pyglet, pygame, pyglm, moderngl_window; print('ok')"`

---

## 🛠️ Build & Packaging

- Normal Python package using flit. Not required for tests or running.
- If needed: `uv build` (or `python -m build`) will produce a wheel/sdist;
  entry points defined in `pyproject.toml`.

---

## 🕹️ Run

- Interactive game window:
    - `uv run riverborn` (entry point: `riverborn.mgl_terrain:main`)
    - or `uv run python run_game.py`
- Headless smoke run is not applicable; tests provide headless rendering.

---

## ✅ Test & Validation (Required Before PR)

- Run all tests headless:
    - `uv run pytest -q`
- Expected behavior as of Aug 2025 on Windows with Python 3.13 and uv:
    - `tests/test_shader.py`: all pass when shader preprocessor path mapping
      matches absolute filenames (see pitfalls below)
    - `tests/test_rendering.py`: uses `moderngl.create_standalone_context` and
      compares rendered images to `tests/reference_*.png`
- If reference PNG mismatch occurs, tests will save a diff image next to the
  references.

---

## 🧹 Lint & Type-Check

- mypy configuration exists in `pyproject.toml` for a few files only.
- Not a required gate for tests; expect many type errors if you run mypy on the
  whole tree. **Do not treat mypy failures as a CI failure unless the user asks.**

---

## ⚠️ Known Pitfalls & Workarounds (Windows Friendly)

- `pip` may be missing in the repo’s `.venv`; prefer `uv`. If you must use the
  venv, run: `python -m ensurepip -U; python -m pip install -r requirements.txt`.
- OpenGL context is required for the interactive game. Tests use a headless
  moderngl standalone context (no visible window or GPU surface required).
- Imageio reads package data using `importlib.resources`. **Always access assets
  via `importlib.resources.files('riverborn')`, not raw relative paths.**
- Shader preprocessor tests expect **full file paths** in source mapping for
  included files and **plain text (no ANSI/color codes)** in format output. If
  you modify `shader.py`, keep these invariants.
- Terrain noise uses perlin-numpy. **Shapes passed to
  `generate_perlin_noise_2d` must be multiples of `res`; `res` should divide
  (`segments+1`).** The tests build with `segments=64`, so `res` must divide 65;
  choose `res=(5,5)` or compute a compatible res from noise_scale.

---

## 🗂️ Project Layout (Where to Change Things)

- **Entry point/main:** `src/riverborn/mgl_terrain.py` (`WaterApp`, `main()`)
- **Scene/renderer core:** `src/riverborn/scene.py` (models, instances, lights,
  shadows), `src/riverborn/shadow.py`, `src/riverborn/shader.py` (loader and
  simple preprocessor), `src/riverborn/terrain.py` (mesh generation helpers)
- **Gameplay/tools:** `src/riverborn/animals.py`, `plants.py`, `tools.py`,
  `picking.py`, `camera.py`, `ripples.py`, `blending.py`, `screenshot.py`
- **Data/assets:** `src/riverborn/shaders/*.glsl|.vert|.frag`,
  `src/riverborn/models/*.obj|.mtl|*.png|*.tga`, `src/riverborn/textures/`,
  `src/riverborn/sounds/`
- **Tests:** `tests/test_shader.py`, `tests/test_rendering.py` with references
  in `tests/*.png`
- **Packaging/config:** `pyproject.toml` (deps, entry points, mypy settings),
  `requirements.txt` (lock exported via uv), `uv.lock`

---

## 🔁 CI & Pre-Commit

- No GitHub Actions are defined in this repo. **Treat running pytest locally as
  the validation gate.**

---

## 🏃‍♂️ Commands That Work Reliably

- `uv sync --python 3.13`
- `uv run -q pytest -q`
- `uv run riverborn` (launch window)
- `uv run python run_game.py` (launch window)

---

## ⏱️ Performance Notes

- pytest completes quickly (<3s) on a typical dev machine when passing.

---

## 📝 Expectations for Changes

- **Keep tests passing.**
- If you touch `shader.py`, ensure:
    - `preprocess_shader` returns `(processed_text, source_locs)` where
      `source_locs` include absolute filesystem paths for include files when
      `include_dirs` are strings or Path-like pointing to the filesystem; and
      preserve the provided `source_filename` for top-level code.
    - `format_shader_with_line_info` must produce plain text **without ANSI
      escapes**; format exactly `<line>  # <filename>:<lineno>` per test
      expectations.
- If you touch `terrain.py`/`scene.py`, ensure noise generation uses a
  shape/res pair where (`segments+1`) is divisible by res components used by
  perlin-numpy. The tests build with `segments=64`, so res must divide 65;
  choose `res=(5,5)` or compute a compatible res from noise_scale.

---

## 🏁 Authoritative Steps — Trust These

- **Always prefer uv for creating/using the environment and running commands.**
- **Always run `uv run pytest -q` before submitting a PR.**
- **Only search the codebase if these instructions are insufficient or you
  detect an inconsistency.**
