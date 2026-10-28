"""
Automation of Shear Test Simulations

This script automates the simulation of the CPB'06 yield criterion and Swift hardening law 
using Abaqus for a shear test. Instead of modeling the geometry and mesh from scratch, 
it opens a pre-existing master CAE model and dynamically updates its material properties. 
The script reads a pre-calculated Design of Experiments (DoE) with varying material parameters 
from an external CSV file, executes the simulation using a user material subroutine (UMAT), 
and extracts specific results—namely the reaction forces from the pulling grips and the 
averaged logarithmic strains from the center of the specimen—directly from the Output 
Database (ODB). Finally, it filters out aborted simulations and saves the valid processed 
data into individual CSV files for further machine learning analysis.
"""

# -*- coding: mbcs -*-
# abaqus cae noGUI=run_doe.py
from abaqus import *
from abaqusConstants import *
import os, csv, time, sys

# ========================================================================= #
# SUPPORT FUNCTIONS
# ========================================================================= #
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def timer_tick(start_time):
    end_time = time.time()
    elapsed_minutes = int((end_time - start_time) // 60)
    elapsed_seconds = int((end_time - start_time) % 60)
    print("Finished in {}:{:02d} minutes.".format(elapsed_minutes, elapsed_seconds))

def limpar_lixo():
    lixo = ['.odb', '.lck', '.sta', '.msg', '.dat', '.com', '.prt', '.sim']
    for ext in lixo:
        if os.path.exists('Job_Alexandre' + ext):
            try: os.remove('Job_Alexandre' + ext)
            except: pass

# ========================================================================= #
# 1. FOLDERS AND PATHS
# ========================================================================= #
current_dir = os.getcwd() 

# Master CAE file name
NOME_FICHEIRO_CAE = 'Shearprobe.cae' 

caminho_cae = os.path.join(current_dir, NOME_FICHEIRO_CAE)
caminho_umat = os.path.join(current_dir, 'UMMDp_FLC.f')
caminho_doe_mestre = os.path.join(current_dir, 'data', 'doe', 'doe_shear.csv')
MYCSVDIR = os.path.join(current_dir, 'data', 'ofat_shear') 
make_dir(MYCSVDIR)

# ========================================================================= #
# 2. OPEN THE MASTER MODEL
# ========================================================================= #
print("Opening the master CAE file...")
openMdb(pathName=caminho_cae)

NOME_MODELO = 'Model-1'

# ========================================================================= #
# 3. SIMULATION LOOP (17 COLUMNS)
# ========================================================================= #
print("Reading the Master DOE Database...")
materiais_virtuais = []
with open(caminho_doe_mestre, 'r') as f_doe:
    leitor = csv.reader(f_doe, delimiter=';')
    next(leitor) 
    for linha in leitor: materiais_virtuais.append([float(x) for x in linha])

MATERIAL_INICIAL = 1099

for index, valor in enumerate(materiais_virtuais, start=1):

    if index < MATERIAL_INICIAL:
        continue

    # READ THE EXACT 17 COLUMNS FROM THE CSV
    v_C11, v_C12, v_C13 = valor[0], valor[1], valor[2]
    v_C21, v_C22, v_C23 = valor[3], valor[4], valor[5]
    v_C31, v_C32, v_C33 = valor[6], valor[7], valor[8]
    v_C44, v_C55, v_C66 = valor[9], valor[10], valor[11]

    v_a = valor[12]
    v_k2006 = valor[13]
    v_sigma0 = valor[14] # sigma0 is at index 14
    v_k = valor[15]      # k is at index 15
    v_n = valor[16]      # n is at index 16

    # Fixed and calculated parameters
    v_E = 210000.0
    v_poisson = 0.3
    
    # CALCULATE e0 IN PYTHON (as it is not in the CSV)
    # Swift Law formula: Y0 = K * (e0)^n  =>  e0 = (Y0/K)^(1/n)
    v_e0 = round(float((v_sigma0 / v_k)**(1.0 / v_n)), 6)

    start_time = time.time()
    print('\n[SHEAR TEST] Sample {} of {}:'.format(index, len(materiais_virtuais)))
    
    # 3.1 Replace constants in the EXACT order required by the UMAT (24 values)
    mdb.models[NOME_MODELO].materials['Material-1'].UserMaterial(mechanicalConstants=(
        0, 0, v_E, v_poisson, 3, 
        v_C11, v_C12, v_C13, v_C21, v_C22, v_C23, v_C31, v_C32, v_C33, v_C66, v_C55, v_C44, 
        v_a, v_k2006, 2, v_k, v_e0, v_n, 0
    ))

    # 3.2 Prepare and run the Job
    job_name = 'Job_Alexandre'
    limpar_lixo()
    if job_name in mdb.jobs.keys(): del mdb.jobs[job_name]
        
    myJob = mdb.Job(name=job_name, model=NOME_MODELO, type=ANALYSIS, userSubroutine=caminho_umat, resultsFormat=ODB, numCpus=1, numDomains=1)
    
    print("  -> Calculating in Abaqus...")
    myJob.submit(consistencyChecking=OFF)
    myJob.waitForCompletion() 
    time.sleep(2) 

    # ================================================================================================= #
    # 4. DATA EXTRACTION 
    # ================================================================================================= #
    odb_Path = job_name + '.odb'
    try:
        shear_ODB = session.openOdb(name=odb_Path)
        frames = shear_ODB.steps['Step-1'].frames
        num_frames = len(frames)
        tempo_final = frames[-1].frameValue
        
        # Check if simulation completed successfully
        if num_frames != 21 or tempo_final < 0.99:
            raise ValueError("Collapsed at {}% (Saved {} frames). Rejected.".format(round(tempo_final*100, 1), num_frames))
            
        nodes_puxa_labels = [node.label for node in mdb.models[NOME_MODELO].rootAssembly.sets['Set-Puxa'].nodes]
        elems_centro_labels = [elem.label for elem in mdb.models[NOME_MODELO].rootAssembly.sets['Set-Centro-Elems'].elements]

        if len(nodes_puxa_labels) == 0 or len(elems_centro_labels) == 0:
            raise ValueError("Empty Sets! Could not find 'Set-Puxa' or 'Set-Centro-Elems'.")

        # CSV FILENAME NOW WITH ALL 17 VARIABLES
        csv_filename = r'{}\{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.csv'.format(
            MYCSVDIR, v_C11, v_C12, v_C13, v_C21, v_C22, v_C23, 
            v_C31, v_C32, v_C33, v_C44, v_C55, v_C66, v_a, v_k2006, v_sigma0, v_k, v_n)

        all_data = []

        for i in range(1, 21):
            rf_values, le_values = frames[i].fieldOutputs['RF'].values, frames[i].fieldOutputs['LE'].values
            soma_forca_z = sum([abs(val.data[2]) for val in rf_values if val.nodeLabel in nodes_puxa_labels])
            
            soma_le11, soma_le22, soma_le12, num_centro = 0.0, 0.0, 0.0, 0
            for val in le_values:
                if val.elementLabel in elems_centro_labels:
                    # [0] is LE11, [1] is LE22, [3] is LE12
                    soma_le11 += val.data[0]; soma_le22 += val.data[1]; soma_le12 += val.data[3]; num_centro += 1

            if num_centro > 0: all_data.append([soma_forca_z, soma_le11/num_centro, soma_le22/num_centro, soma_le12/num_centro])
            else: all_data.append([soma_forca_z, 0.0, 0.0, 0.0])

        shear_ODB.close()
        
        with open(csv_filename, 'wb') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Forca_Z', 'LE11', 'LE22', 'LE12'])
            writer.writerows(all_data)

        print("  ✅ PERFECT CSV! 20 rows saved.")
        timer_tick(start_time)

    except Exception as e:
        print("  ❌ Error: {}".format(e))
        try: shear_ODB.close()
        except: pass
        timer_tick(start_time)
        continue
