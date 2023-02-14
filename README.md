# AI_IIT105_Task_Water_Potability
## Water Potability Classification
Binary classification problem in which, based on the values of nine different attributes/features, we classify water samples into one out of two categories (0-non potable, 1-potable). The following table contains the aforementioned features as well as a basic description for each one of them.

| # | Attribure       |  Description                         |               
|---|-----------------|--------------------------------------|
| 0 | pH              | acid–base balance of $H_{2}O$ [0-14] |
| 1 | Hardness        | Ca & Mg content in $H_{2}O$ [mg/L]   |
| 2 | Solids          | Total dissolved solids (TDS) [ppm]   |
| 3 | Chloramines     | Total dissolved $NH_{2}Cl$ [ppm]     |
| 4 | Sulfate         | $SO_{4}^{2-}$ content in $H_{2}O$ [mg/L] |
| 5 | Conductivity    | Electrical Conductivity of  $H_{2}O$ [μS/cm] |
| 6 | Organic_Carbon  | Concentration of Organic Carbon [ppm] |
| 7 | Trihalomethanes | Chemicals found in water treated with chlorine [μg/L] |
| 8 | Turbidity       | Light emitting property of water [NTU] |

## Basic Code Information
The code has been developped in **python 3.8.2**. To run the code, the requirements.txt file must be used in order to load all the needed modules/packages. Then, using an IDE (i.e spyder), we can run the code by executing the main.py file which is located inside the src folder.

## Repository Structure
The repository structure is simple and self-explanatory. It containts the following folders and files:

**requirements.txt** - File that contains all the modules/packages information needed to run the code.

**Presentation folder** - Contains the presentation both in .pptx and .pdf format.

**Report folder** - Contains the report as .pdf file. Also, there is a compressed file that contains the LaTeX code that has been created.

**src folder** - contains the following files and folders
| Files/Folders   |  Description                         |               
|-----------------|--------------------------------------|
| main.py         | Main file of our source code |
| Results         | Folder that contains all the results that the code produces (.dat and .json) |
| import_data     | Folder that contains the water_potability.csv file (data used as input for training/test) |
| import_pys      | Code has been splitted in multiple files for better overview. This folder contains all the .py files apart from the main.py |

Files and the dependencies between them can be seen in the following PlantUml diagram:

![alt text](https://github.com/NMech/AI_IIT105_Task_Water_Potability/blob/main/plantUml.PNG?raw=true)

## Basic Results
Below we can see the results of our trained models. More details can be found in the report.
| Classifier        | Accuracy | Precision | Recall | f1    |
|-------------------|----------|-----------|--------|-------|
| Gradient Boosting | 0.677    | 0.654     | 0.279  | 0.391 |
| Random Forest     | 0.686    | 0.698     | 0.275  | 0.394 |
| RBF SVM           | 0.692    | 0.684     | 0.320  | 0.436 | 
| Voting Classifier | 0.684    | 0.683     | 0.283  | 0.400 |
