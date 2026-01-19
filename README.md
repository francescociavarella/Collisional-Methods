# Collisional_Methods_for_Coding
Simulation of **Excitonic transport** dynamics in a **excitonic dimer** using a stochastic unraveling approach based on **Collisional Methods**, both in **Quantum Jump** and **Difussive Limits**.
The results are validated through a comparison with the dynamics derived from the **Lindblad Master Equation**. Furthermore, the collisional evolution is benchmarked against the **Isolated System** dynamics (i.e., purely unitary evolution without the collisional interaction).

Author : Francesco Ciavarella - francesco.ciavarella@studenti.unipd.it

---

## Contents : 

1. [Initial Configuration](#initial-configuration)
2. [Repository Organization](#repository-organization)
3. [Future Goals](#future-goals)

---

## Initial Configuration
To replicate the results clone the repository and recreate the Conda enviroment, to ensure all required libraries are installed.

1. **Clone the Repository**
    ```
    git clone https://github.com/francescociavarella/Collisional_Methods_for_Coding.git
   cd Collisional_Methods_for_Coding
    ```
2. **Create the Enviroment wit Conda**
    ```
    conda env create -f environment.yml
    ```
3. **Activate Enviroment**
    ```
    conda acvtivate mc
    ```
Now you are ready to navigate the repository and run the notebooks.

---

## Repository Organization

This repository is organized in three main sections:

* **Codes** : contains the source code for the project.
  * `Main_Dynamics.py`: implements the core system evolution
  * `Plot.ipynb`: for visualizing results via standard 2D plotting
  * `Bloch_Sphere.ipynb`: for 3D visualizations of the state's dynamics on the Bloch's Sphere.

* **Results** : contains some example of simulation results. 

* **Documentation** : provides the theoretical background.
  * `Report.pdf`: a brief introduction to *Collisional Methods*,  the physical basis of the main program and its structure.
  * `Demonstration.pdf`: mathematical derivation of some of the most important equations used.

---

## Future Goals

Currently the focus is on the investigation and validation of an hypothetical **Intermediate Limit**, between the two regimes already analized (i.e. Quantum Jump and Diffusive); the goal is characterized the resulting unraveling and its physical dependencies.

Future work will be the investigation of a **more complex dynamics and system topology**, for example using more than two site or introducing other different Enviroment effects (beyond the pure Dephasing currently studied)

---

