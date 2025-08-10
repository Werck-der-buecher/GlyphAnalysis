# Analysis of skeletonized glyphs
Repository that contains code and jupyter notebooks to compare glyph collections between incunable reproductions. This core requires previously skeletonized glyphs. Please have a look at the other Werck der buecher projects that allow you to extract glyphs from historical prints and predict a corresponding skeleton representation.

## Installation

#### 1. Install miniconda: 

   https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

#### 2. Clone the repository:
To get started, simply clone the repository to your computer. Open a terminal (Command Prompt, PowerShell, or a terminal window on Mac/Linux) and run the following command:

```bash
git clone git@github.com/Werck-der-buecher/GlyphAnalysis.git
```

This command will clone the main project repository and automatically download and set up the AdaAttN submodule in the current version.

#### 3. Navigate to main project folder

```bash
cd GlyphAnalysis
```

#### 4. Create a new conda environment, activate it, and install the following packages:

```bash
conda env create -n wdb_skelanalysis
conda activate wdb_skelanalysis

pip install numpy matplotlib pillow seaborn pandas ipycanvas umap-learn pystackreg hdbscan plotly jupyter ipywidgets jupyter_contrib_nbextensions POT
conda install -n base -c conda-forge jupyterlab_widgets
```

## Execution of interactive Jupyter Notebook

The analysis code is provided as a Jupyter Notebook. To start this notebook, make sure that you have activated the above conda environment. Then, call the following command, which opens up a new window in your standard browser application.

```bash
jupyter lab ./skeleton_analysis.ipynb
```

Alternatively, you can run the notebook in VSCode or other IDEs of your choice.