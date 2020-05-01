

As a demo, here we try to reproduce the black learning curve displayed in Figure S11B of reference 3 (i.e., the amons paper, see `$AQML_ROOT/doc/amons.pdf`, where `AQML_ROOT` is the root directory of `aqml` code).

- prerequisite: `orca4` has to be acquired (One has to fill the registration form at https://cec.mpg.de/orcadownload/ before obtaining a copy).

- python scripts to be used (located under `$AQML_ROOT/bin`): 

  - `gen_orca_jobs`: generate orca4 input files 
    - e.g., `gen_orca_jobs -loose -t optg -m b3lyp -b vdz -n 1 01.sdf`
 
  - `batch_orca`: run orca4 jobs 
 
  - `orca2xyz`: read geometry and molecular property from orca4 output file
 
  - `aqml`: a script for training/test using amons-based machine learning (AML) model


- Timings 
  - Generation of amons: ~10^0 cpu seconds
  - DFT calculation: ~10^0 cpu minutes per amons, ~10^2 cpu minutes for target
  - AML: ~10^0 cpu seconds

