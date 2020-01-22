

Details about the files in this folder:

- `g7v/`: the directory containing all amons ($N_I\leq 7$) , including both covalent and vdW amons. Single point energies calculated at the level of PBE/Def2-SVP are to be used for energy prediction of 1ubq calculated at the same level.

- `amons-r3.6`: cutout amons used for prediction of properties of long-range nature, i.e., NMR shifts.


QML results are shown below:

- for energy prediction, run the following command
```bash
aqml -p pbedef2svp -train g7v -test target 
```
gives the following results:
```bash
...
   1     51          nan          nan
   2    412    8585.7105    8585.7105  (DressedAtom mae=  10761.7528)
   3    858     351.8642     351.8642  (DressedAtom mae=   3837.3767)
   4   1912     495.7883     495.7883  (DressedAtom mae=   3575.1935)
   5   3603     152.7714     152.7714  (DressedAtom mae=   3004.2573)
   6   5866     171.0327     171.0327  (DressedAtom mae=   2974.6095)
   7   9305       7.6426       7.6426  (DressedAtom mae=   2589.1867)
 elapsed time:  10306.493880271912  seconds
```
where column 1-5 indicate `N_I` (the number of heavy atom), `MAE` (mean absolute error), `RMSE` (root mean squared error), prediction error of dressed atom approximation. `nan` corresponds to invalid prediction.


