# MasterThesis

###### Disclaimer: The HotSpot thermal simulator found in the folder "CustomHotSpot" is a slighlty modified version of HotSpot which can be found here: https://github.com/uvahotspot/HotSpot. No modifications are made in the functionality, only to the amount of output provided. (The License of HotSpot can also be found in that folder under "LICENSE")


## Requirements to run
- Ubuntu 20.04 (or newer)
- Currently HotSpot is compiled for x86 - 64bit machines. For different machines, you may have to recompile following the instructions here: https://github.com/uvahotspot/HotSpot/wiki/Getting-Started (and you have to compile HotSpot in the `CustomHotSpot` folder.)
- Install all dependencies using: `sudo apt install libblas-dev libsuperlu5 libsuperlu-dev`
- Python 3.10.12 (or newer)
- Dependencies for Python in the `requirements.txt` file, which can be installed using: `pip install -r requirements.txt`
- A valid Gurobi License (visit https://www.gurobi.com for more information)
- Jupyter Notebook to open the `.ipynb` files (https://jupyter.org/install)

## How to do things
### Compute Thermal Coefficient Matrix
To compute the Thermal Coefficient Matrix used in the Chapter Matrix Model, consult `GenerateThermalModelParameters.ipynb`.
Define the Processors dimensions in the first block, and run the tiles, whereafter in the `HotSpotConfiguration` folder the matrix is written to a file called `thermal_coefficients.txt` in the processors folder.

### Experiments
The experiments can be found in the respective `.ipynb` files.
- For Matrix Model (TATS-ND-FP): See `MatrixModelFormulations.ipynb`
- For Power Levels (TATS-ND-k): See `PowerLevelFormulations.ipynb`
- For Restricted Assignment (RA-BC): See `RestrictedAssignmentFormulations.ipynb`

In the first block of the Experiments the processors dimensions can be defined.

A number of processor dimenstions are predefined, namely:
- 2 x 2 x 2
- 2 x 2 x 3
- 3 x 3 x 2
- 3 x 3 x 3

More processor configurations can be defined manually, in the `HotspotConfiguration` folder.
