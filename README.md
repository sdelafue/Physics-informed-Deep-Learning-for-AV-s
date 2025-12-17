# PIDL for AV's Project

This project uses the [Julia Programming Language](https://julialang.org/). We use a local environment to ensure package reproducibility across the team.

## Prerequisites

Before starting, ensure you have the following installed:
1. **Julia** (Latest stable release): [Download Here](https://julialang.org/downloads/)
2. **VS Code**: [Download Here](https://code.visualstudio.com/)
3. **Julia Extension for VS Code**: Search for "Julia" in the VS Code Extensions marketplace and install the one by *Julia Computing*.

## Environment Setup

To ensure you are using the exact package versions defined in `Manifest.toml`, follow these steps:

1. **Open the Project:**
   Clone this repository and open the root folder in VS Code.

2. **Start the Julia Terminal:**
   Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) and type:
   > `Julia: Start REPL`

3. **Activate the Environment:**
   In the Julia terminal (at the bottom), perform the following:
   * Press `]` to switch to the Package Manager mode. (The prompt will change to `(@v1.x) pkg>`).
   * Type the following command to activate the project folder:
     ```julia
     activate .
     ```
   * *Check:* The prompt should change to `(MyJuliaRover) pkg>`.

4. **Install Dependencies:**
   While still in Package Manager mode, run the instantiation command. This downloads all required packages listed in the `Manifest.toml` file:
   ```julia
   instantiate
   ```

## Running the nuScenes Parser

The `nuscenes_parser.jl` script processes LiDAR data from the nuScenes dataset and generates Bird's Eye View (BEV) occupancy grids.

### Setup: Add Dataset Files

Before running the script, you need to add the nuScenes LiDAR data:

1. Download the nuScenes dataset (v1.0-mini or full version)
2. **Copy all `.pcd.bin` files** into the `datasets/LIDAR_TOP` folder in this repository
3. The folder structure should look like:
   ```
   project-root/
   ├── datasets/
   │   └── LIDAR_TOP/
   │       ├── n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
   │       ├── n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597423.pcd.bin
   │       └── ... (more .bin files)
   ```

**Note:** The `.pcd.bin` files are not included in the repository due to their size.

### Option 1: Auto-Detection (Easiest)

The script will automatically search for the dataset in common locations or prompt you to enter the path.

1. **Open Terminal/Command Prompt:**
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - Mac: Press `Cmd + Space`, type "Terminal", press Enter
   - Linux: Press `Ctrl + Alt + T`

2. **Navigate to the Project Folder:**
   ```bash
   cd path/to/project
   ```

3. **Run the Script:**
   ```bash
   julia nuscenes_parser.jl
   ```

4. **If Prompted:** Enter the full path to your `LIDAR_TOP` folder when asked (no quotes needed).

### Option 2: Provide Path Directly

You can pass the dataset path as a command-line argument:

```bash
julia nuscenes_parser.jl "C:/path/to/v1.0-mini/samples/LIDAR_TOP"
```

**Important Notes:**
- **Use quotes** if your path contains spaces (e.g., `"C:/My Files/dataset/LIDAR_TOP"`)
- The path should point to the `LIDAR_TOP` folder containing `.pcd.bin` files, not to a specific file
- You can use either forward slashes (`/`) or backslashes (`\`) - the script handles both

### Example Paths:
```bash
# Windows
julia nuscenes_parser.jl "C:/Users/YourName/Downloads/v1.0-mini/samples/LIDAR_TOP"

# Mac/Linux
julia nuscenes_parser.jl "/Users/YourName/Downloads/v1.0-mini/samples/LIDAR_TOP"
```

### Troubleshooting:
- **Path not found?** Make sure you're pointing to the `LIDAR_TOP` folder, not its parent directory
- **Spaces in path?** Remember to wrap the entire path in quotes
- **No .bin files found?** Verify you've unzipped the dataset and the folder contains `.pcd.bin` files