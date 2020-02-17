

As a demo, here we try to reproduce the learning curve displayed in Fig. 

- prerequisite: a copy of the quantum chemistry program `orca4' has to be acquired (for downloads, fill the registration form at https://cec.mpg.de/orcadownload/).

- python scripts to be used (located under `$AQML_ROOT/bin`): 

  - gen_orca_jobs: generate orca4 input files 

  - batch_orca: run orca4 jobs 

  - orca2xyz: read geometry and molecular property from orca4 output file

```bash
gen_orca_jobs -loose -t optg -m b3lyp -b vdz -n 1 g7/*.sdf target/*.sdf
```
The option `-n 1` specifies the number of processes to be used. Choose a larger number to speed up computations.

  - Then run orca4 jobs through calling the script `batch_orca`:
```bash
batch_orca g7/*.com target/*.com >out1 &
```
Note: before running, the user needs to reset the path to `orca4` in file `which batch_orca_base` (located under `$AQML/bin`, where `$AQML` denotes the root directory of the `aqml` package).

  - Convert output files to xyz format once all calculations are done:
```bash
orca2xyz g7/*.out target/*.out
```
The resulting `xyz` files contain relaxed geometries, the same as the usual `xyz` file format-wise. Poperties (only energy as the default case) is written to the second line, with the format of, say `b3lypvdz=-40.0321`, which means the single point energy calculated at the level B3LYP/cc-pVDZ is -40.0321 Hartree (Atomic unit is the default for all properties in `xyz` files).


- Timings 
  - generation of amons: several seconds
  - DFT calculation: several cpu minutes per amons, several cpu minutes to several hours for query
  - AML: of the order seconds for small target, 

