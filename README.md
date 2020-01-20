# aqml
Amons-based quantum machine learning for chemistry problems

Features:
1) Parameter-free SLATM representations, and its local (atomic) counterpart.
   * A new pair-wise version is available, dealing with dataset consisting of many elements.
2) BAML representation, including up to 4-body potential (force-field inspired)
3) KRR, multi-fidelity KRR
4) Automatic sampling of training set that are most representative for any query molecule (composition and size-independent. So far, limited to molecules without periodic boundary condition).


Todo's
1) Distored configuration generation on-the-fly for MD 
   * Use SLATM-derived metric
2) force prediction using SLATM
   * geometry optimization, molecular dynamics
  
Usage
1) bin/aqml: 

