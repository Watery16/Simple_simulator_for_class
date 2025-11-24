# Simulator for Two-Mirror / Two-Pinhole Alignment (Optiland)

This repository uses [Optiland](https://github.com/HarrisonKramer/optiland) for differentiable ray tracing of a two-mirror, two-pinhole alignment system. It includes utilities for irradiance computation, signal scanning, gradient-based optimization, visualization, and experiment logging.

## Dependencies
- Python 3.10+
`pip install -r requirements.txt`

Example install (pick the torch build for your system):
```bash
pip install optiland==0.5.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib vtk wandb
```

## Project Layout
- `Simulator/Optical_System/Optical_Sys.py`: Core optical system builder (two mirrors, two pinholes, detectors) with helper methods to set mirror angles and pinhole sizes.
- `Simulator/utils/`
  - `computing.py`: Irradiance maps, circular masks, centroid, power integration, detector edges, plotting helper.
  - `loss.py`: Losses based on irradiance MSE, power, and centroid error; combined loss helpers.
  - `helper.py`: Rotation matrices, mirror normals, reflection geometry, and position calculations.
  - `visualize.py`: Off-screen 3D rendering and replay of optimization trajectories with irradiance maps.
  - `logger.py`: CSV logger with optional Weights & Biases integration.
- Top-level notebook show usage examples.

## Contact 
If you have any question or there are any bugs, please contanct swllen25@gmail.com

