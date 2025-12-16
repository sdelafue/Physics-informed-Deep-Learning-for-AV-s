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
