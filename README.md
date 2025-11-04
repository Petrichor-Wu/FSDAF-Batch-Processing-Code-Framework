Of course. Here is the professional English version of the README for your FSDAF Batch Processing Code Framework.

-----

# 🚀 FSDAF Batch Processing Code Framework

### **Project Overview**

This project provides a **stable, efficient, and thoroughly debugged** code framework for the FSDAF (Flexible Spatio-temporal Data Fusion) algorithm. It is a comprehensive **toolset**, consisting of multiple Python scripts and configuration files, designed specifically for the matching, fusion, and preprocessing of time-series remote sensing data, optimized using NDVI data as the primary example.

The framework focuses on resolving common issues found in the original FSDAF code, such as **index out-of-bounds errors**, **file I/O conflicts**, and **data type inconsistencies**, thereby ensuring stability and reliability for large-scale data processing tasks.

-----

### ✨ **Key Features and Improvements**

| Module | Core Functionality | Key Fixes and Enhancements |
| :--- | :--- | :--- |
| **Stability** | FSDAF Core Algorithm | **Comprehensive array boundary fixes** (`array_bounds_fix.py` / `fsdaf_core.py`) to completely eliminate runtime index out-of-bounds errors. |
| **Automation** | Batch Processing Logic | `batch_fsdaf.py` implements **automatic matching** of coarse/fine resolution image pairs based on filename timestamps, facilitating **parallel processing**. |
| **I/O Optimization** | Raster Read/Write Utilities | Explicit closure of file handles (`utils.py`) to resolve file locking issues common in Windows/Linux environments, improving batch stability. |
| **Data Quality** | Data Preprocessing | **Anomaly correction** (high values $\to 9999$; non-NoData low values $\to -2000$) and **NoData unification** ($\to -99999.0$), ensuring high-quality input for fusion. |
| **Classification** | ISODATA | Uses a customized `myISODATA` algorithm (`isodata.py`), with configurable parameters, for land cover classification. |
| **Flexibility** | Configuration Management | All parameters and paths are centrally managed via `.yaml` files (`batch_config.yaml`, `parameters_fsdaf.yaml`). |

-----

### 📁 **File Structure**

| Filename | Description |
| :--- | :--- |
| `batch_fsdaf.py` | **Main Execution Script.** Reads configuration, automatically matches image combinations, and manages parallel processing tasks. |
| `fsdaf_core.py` | **FSDAF Core Algorithm Implementation.** Contains the fusion logic, block processing, and calls to boundary fix routines. |
| `array_bounds_fix.py` | **Index Fix Module.** Provides safe index checking and array boundary correction functions for use by the core algorithm. |
| `utils.py` | **Utility Script.** Contains fixed raster read/write functions (I/O fixes) and geo-reference copying utilities. |
| `isodata.py` | **Classification Module.** Implements the `myISODATA` algorithm for land cover classification. |
| `batch_config.yaml` | **Batch Configuration.** Defines input/output folder paths and multiprocessing settings. |
| `parameters_fsdaf.yaml` | **Algorithm Parameters.** Defines FSDAF core parameters (e.g., window size, number of similar pixels) and ISODATA classification settings. |

-----

### 🛠️ **Quick Setup and Execution**

#### **1. Prerequisites**

The project relies on standard remote sensing and scientific computing libraries. Ensure the following dependencies are installed in your environment:

```bash
# Core Dependencies
pip install numpy pyyaml gdal rasterio scipy fiona
# Note: The code utilizes osgeo.gdal directly; ensure your gdal installation is correct.
```

#### **2. Configure Paths (`batch_config.yaml`)**

Open the `batch_config.yaml` file and modify the following crucial paths to match your data storage locations:

| Parameter | Description | Example Value |
| :--- | :--- | :--- |
| `coarse_prediction_dir` | Directory for the Coarse (low-resolution) image at prediction time $T_p$. | `E:/Data/NDVI/MCD19A3CMG/Processed` |
| `fine_first_pair_dir` | Directory for the Fine (high-resolution) image at the base time $T_1$. | `E:/Data/NDVI/MOD13Q1/Fine_Processed` |
| `coarse_first_pair_dir` | Directory for the Coarse image at the base time $T_1$. | `E:/Data/NDVI/MOD13Q1/Coarse_Processed` |
| `output_dir` | Output directory for the final fused results. | `E:/Data/NDVI/FSDAF_Results` |

#### **3. Configure Algorithm Parameters (`parameters_fsdaf.yaml`)**

Open the `parameters_fsdaf.yaml` file and adjust the algorithm parameters based on your data characteristics (e.g., resolution, DN value range) and experimental needs, paying close attention to:

  * `w` (Local Window Size)
  * `num_similar_pixel` (Number of similar pixels)
  * `DN_min` / `DN_max` (Valid data range)
  * `background` (NoData fill value)
  * ISODATA classification parameters (`I`, `maxStdv`, `minDis`)

#### **4. Execute the Batch Process**

Run the main script from your command line, passing the configuration file path as an argument:

```bash
python batch_fsdaf.py /path/to/your/batch_config.yaml
```

The script will automatically identify matching image pairs, initiate parallel processing for concurrent computation, and write the final fused results to the designated `output_dir`.
