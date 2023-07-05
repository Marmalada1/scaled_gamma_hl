# This program calculates the scaled gamma created by Hüllermeier and Henzgen.

To install the dependencies do following steps:

## 1. create requirements file:
```
pip install pipreqs

pipreqs /path/to/project
```
## 2. install dependencies from requirements.txt:
```
pip install -r requirements.txt
```
this program can only be run on python versions **below 3.11.0** because of the library **"numba"**.

To perform a calculation run the function ```scaled_gamma(x,y)``` with x and y being the two list to compare. For a more detailed informations about the function you can look at my bachelor thesis 
"Implementation and experimental investigation of a weighted rank correlation measure"
