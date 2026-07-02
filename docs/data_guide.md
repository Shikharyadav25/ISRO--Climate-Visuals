# India Climate Digital Twin: Data Engineering and Dataset Specification

This document details the data schemas, binary formats, georeferencing coordinate systems, and processing pipelines used to compile the multi-dimensional gridded reanalysis archive for the digital twin platform.

---

## Observational Dataset Configurations

The platform assimilates daily gridded meteorological observations from the India Meteorological Department (IMD Pune) and space-based remote sensing datasets.

| Dataset Key | Variable Name | Grid Dimensions | Spatial Resolution | Spatial Bounding Box (Lat/Lon) | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| IMD Daily Rainfall | `rainfall` | 129 × 135 | 0.25° × 0.25° | Lat 6.5°N - 38.5°N \| Lon 66.5°E - 100.0°E | Binary (.grd) |
| IMD Daily Max Temp | `max_temp` | 31 × 31 | 1.0° × 1.0° | Lat 7.5°N - 37.5°N \| Lon 67.5°E - 97.5°E | Binary (.grd) |
| IMD Daily Min Temp | `min_temp` | 31 × 31 | 1.0° × 1.0° | Lat 7.5°N - 37.5°N \| Lon 67.5°E - 97.5°E | Binary (.grd) |
| INSAT-3D/3DR LST | `lst` | Dynamic | 0.1° × 0.1° | Bounded satellite swath | HDF5 (.h5) |
| INSAT-3D/3DR Rain | `rain` | Dynamic | 0.1° × 0.1° | Bounded satellite swath | HDF5 (.h5) |

---

## Raw Binary File Specifications

IMD gridded data is distributed as flat, unformatted binary files containing single-precision floats (4 bytes per element).

### 1. Precipitation Binary Specification (.grd)
* **File Shape:** 129 columns × 135 rows = 17,415 elements.
* **Array Ordering:** Row-major (C-style).
* **Missing Value Flag:** Coordinates containing values of `99.9` or `-99.9` represent geographical areas outside the Indian landmass (or missing telemetry) and are masked to `NaN`.
* **Geographical Mapping:**
  * Latitude: 135 grid points starting from 6.5°N with a spacing of 0.25°:
    $$\text{Lat}_i = 6.5 + i \cdot 0.25 \quad (i = 0, \dots, 134)$$
  * Longitude: 129 grid points starting from 66.5°E with a spacing of 0.25°:
    $$\text{Lon}_j = 66.5 + j \cdot 0.25 \quad (j = 0, \dots, 128)$$

### 2. Temperature Binary Specification (.grd)
* **File Shape:** 31 columns × 31 rows = 961 elements.
* **Array Ordering:** Row-major (C-style).
* **Missing Value Flag:** Coordinates containing `-999.0` represent invalid or missing grid points and are masked to `NaN`.
* **Geographical Mapping:**
  * Latitude: 31 grid points starting from 7.5°N with a spacing of 1.0°:
    $$\text{Lat}_i = 7.5 + i \cdot 1.0 \quad (i = 0, \dots, 30)$$
  * Longitude: 31 grid points starting from 67.5°E with a spacing of 1.0°:
    $$\text{Lon}_j = 67.5 + j \cdot 1.0 \quad (j = 0, \dots, 30)$$

---

## Data Ingestion and Compilation Pipeline

The data engineering pipeline is executed via `scripts/download_and_decode_all_real.py`, performing the following processing sequence:

```text
[IMD Daily Scraper] --> [Flat Binary Read] --> [NaN Masking] --> [regrid_coarsen] --> [NetCDF4 Append]
```

### 1. Ingestion and Verification
The script calculates the target date range and queries the IMD HTTP servers. Files are saved in `data/raw/` and validated against nominal size thresholds (e.g., exactly 69,660 bytes for precipitation grids).

### 2. NetCDF4 Grid Compilation
Individual daily grids are stacked along the time dimension into CF-compliant NetCDF4 archive files. The netCDF variable properties are configured with standard metadata tags:

```python
# Metadata Variable Definition Example
variable.units = "mm/day"
variable.long_name = "IMD daily gridded precipitation"
variable.missing_value = np.nan
```

### 3. Spatial Resolution Resampling
To perform spatial matrix calculations between the 0.25° rainfall grid and the 1.0° temperature grid, the pipeline uses nearest-neighbor interpolation to broadcast boundaries cleanly:
$$\text{TargetGrid}(x, y) = \text{SourceGrid}(\text{nearest}(x), \text{nearest}(y))$$
This preserves sharp boundaries (e.g., administrative lines or coastlines) compared to bilinear interpolation, which causes blurring at the edges.
