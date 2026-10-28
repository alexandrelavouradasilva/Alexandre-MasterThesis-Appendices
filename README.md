# Alexandre-MasterThesis-Appendices

# Supplementary Digital Appendices for Alexandre Lavoura da Silva's Master's Thesis

**Author:** Alexandre Lavoura da Silva  
**Institution:** University of Aveiro (DETI)  
**Year:** 2026

## Overview
This repository contains the digital appendices accompanying my Master's Thesis. To comply with formatting guidelines and page limits while ensuring complete methodological transparency and reproducibility, extended mathematical deductions, the full submitted manuscript, and all computational scripts are hosted here.

This research focuses on addressing the machine learning challenges and parameter identifiability issues associated with the calibration of the CPB'06 yield criterion, culminating in the projection of the parameter identification problem into its canonical space.

## Repository Structure & Thesis Mapping

* 📂 **[01_Mathematical_Deductions](./01_Mathematical_Deductions)** *(Corresponds to Appendix A)*
  * Contains the step-by-step mathematical expansion of the macroscopic yield condition (CPB'06) for the specific quadratic case ($a=2, k=0$).

* 📂 **[02_Submitted_Manuscripts](./02_Submitted_Manuscripts)** *(Corresponds to Appendix B)*
  * Includes the full PDF of the article *"Machine Learning Challenges in the Calibration of the CPB'06 Yield Criterion"*, submitted to WCCM/ECCOMAS 2026.

* 📂 **[03_Python_Scripts](./03_Python_Scripts)** *(Corresponds to Appendix C)*
  * Contains all Python automation and machine learning source code. This includes the finite element simulation automation scripts for Abaqus (Cruciform, Shear, and Asymmetric tests), as well as the complete PyTorch/Optuna and XGBoost pipelines for parameter identification, physical loss functions, and macroscopic validation. 
  * 📝 *Note: Please refer to the dedicated `README.md` inside this folder for a detailed, script-by-script index of the workflows.*

## Software & Dependencies
To ensure reproducibility, the scripts and simulations in this repository were developed and tested using the following environment:
* **Finite Element Solver:** SIMULIA Abaqus (version 2019)
* **Programming Language:** Python 3.13.5
* **Key ML/Data Libraries:** PyTorch, XGBoost, Optuna, scikit-learn, pandas, numpy

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Citation
If you utilize the code, data, or mathematical deductions provided in this repository for your own research, please consider citing the associated Master's Thesis. 

## Contact
For any questions regarding the methodology, code, or data, please feel free to reach out via [(https://www.linkedin.com/in/alexandre-silva-1587552a1/)].
