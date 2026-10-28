"""
Automation of Asymmetric Notched Tensile Test Simulations

This script automates the simulation of the CPB’06 yield criterion and Swift hardening law 
using Abaqus for an asymmetric notched tension test. Rather than generating the geometry 
dynamically, it loads a pre-existing master CAE model and programmatically updates the 
material parameters by reading an external Design of Experiments (DoE) CSV file. 

To facilitate processing large datasets, the script features a batch control mechanism 
that allows slicing the DoE for parallel or segmented execution. Following the finite 
element simulation using a user material subroutine (UMAT), it systematically extracts 
the overall reaction forces alongside the averaged logarithmic strains from three distinct 
regions of interest (Left, Right, and Center) directly from the Output Database (ODB). 
Finally, incomplete simulations are filtered out, and the extracted tensor fields are 
saved into formatted CSV files to serve as rich kinematic training data for the neural 
network models.
"""

# -*- coding: mbcs -*-
from abaqus import *
from abaqusConstants import *
import os, csv, time, numpy as np

# ========================================================================= #
# CONFIGURATIONS
# ========================================================================= #
NOME_FICHEIRO_CAE = 'notched_simulation.cae'
NOME_MODELO = 'Model-1'
SETS = {'Esq': 'Set-Esq', 'Dir': 'Set-Dir', 'Centro': 'Set-Centro'}
SET_PUXA = 'Set-Puxa' 
MYCSVDIR = os.path.join(os.getcwd(), 'data', 'ofat_assimetrico')
if not os.path.exists(MYCSVDIR): os.makedirs(MYCSVDIR)

# --- NEW: BATCH CONTROL (Slice the DOE) ---
START_INDEX = 1  # Starts at material 3500 (example)
END_INDEX = None    # Put a number (e.g., 5000) to stop there. None goes to the end.
# ----------------------------------------------

def timer_tick(start_time):
    end_time = time.time()
    elapsed_minutes = int((end_time - start_time) // 60)
    elapsed_seconds = int((end_time - start_time) % 60)
    print("Finished in {}:{:02d} minutes.".format(elapsed_minutes, elapsed_seconds))

def limpar_lixo(nome_do_job):
    # Close any open ODBs before attempting to delete files
    for nome_odb in session.odbs.keys():
        session.odbs[nome_odb].close()
        
    extensoes = ['.odb', '.lck', '.sta', '.msg', '.dat', '.com', '.prt', '.sim', '.inp', '.ipm', '.txt', '.log']
    for ext in extensoes:
        ficheiro = nome_do_job + ext
        if os.path.exists(ficheiro):
            try: os.remove(ficheiro)
            except OSError: pass

# ========================================================================= #
# AUTOMATED EXECUTION LOOP
# ========================================================================= #
materiais = []
with open(os.path.join(os.getcwd(), 'data', 'doe', 'DOE_Assimetrico.csv'), 'r') as f:
    leitor = csv.reader(f, delimiter=';')
    for i, linha in enumerate(leitor):
        try:
            materiais.append([float(x) for x in linha])
        except ValueError:
            continue

openMdb(pathName=NOME_FICHEIRO_CAE)

# Slice the materials list to the desired batch
fatia_materiais = materiais[START_INDEX-1:END_INDEX] if END_INDEX else materiais[START_INDEX-1:]

for index, valor in enumerate(fatia_materiais, start=START_INDEX):
    start_time = time.time()
    print('\n[SIMULACAO] Material {} de {}:'.format(index, len(materiais)))
    
    # 1. Define UMAT constants
    v_E, v_poisson = 210000.0, 0.3
    
    sigma0_val = valor[14]
    k_val      = valor[15]
    n_val      = valor[16]
    
    e0_val = (sigma0_val / k_val) ** (1.0 / n_val)
    
    novas_constantes = (
        0, 0, v_E, v_poisson, 3, valor[0], valor[1], valor[2], valor[3], valor[4], 
        valor[5], valor[6], valor[7], valor[8], valor[11], valor[10], valor[9], 
        valor[12], valor[13], 2, k_val, e0_val, n_val, 0
    )
    
    try:
        mdb.models[NOME_MODELO].materials['Material-1'].userMaterial.setValues(mechanicalConstants=novas_constantes)
    except AttributeError:
        mdb.models[NOME_MODELO].materials['Material-1'].UserMaterial(mechanicalConstants=novas_constantes)

    # 2. Run Job
    job_name = 'Job_Mat_{}'.format(index)
    limpar_lixo(job_name)
    
    if job_name in mdb.jobs.keys(): 
        del mdb.jobs[job_name]
        
    myJob = mdb.Job(name=job_name, model=NOME_MODELO, type=ANALYSIS, userSubroutine='UMMDp_FLC.f')
    myJob.submit(consistencyChecking=OFF)
    myJob.waitForCompletion()

    print(" -> Final Job Status: {}".format(myJob.status))

# ========================================================================= #
# 3. DATA EXTRACTION (FORCES AND STRAINS ONLY)
# ========================================================================= #
    try:
        if not os.path.exists(job_name + '.odb'):
            raise IOError("The ODB file was not generated on disk.")
            
        odb = session.openOdb(job_name + '.odb')
        frames = odb.steps['Step-1'].frames
        num_frames = len(frames)
        tempo_final = frames[-1].frameValue
        
        # Check if simulation completed successfully
        if num_frames != 21 or tempo_final < 0.99:
            raise ValueError("Collapsed at {}% (Saved {} frames). Rejected.".format(round(tempo_final*100, 1), num_frames))
        
        nodes_puxa = mdb.models[NOME_MODELO].rootAssembly.sets[SET_PUXA].nodes
        nodes_puxa_labels = [n.label for n in nodes_puxa]
        
        elems_por_set = {}
        for k, set_name in SETS.items():
            elementos = mdb.models[NOME_MODELO].rootAssembly.sets[set_name].elements
            elems_por_set[k] = [e.label for e in elementos]

        valor = [round(x,4) for x in valor] 

        nome_csv = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
            valor[0], valor[1], valor[2], valor[3], valor[4], valor[5],
            valor[6], valor[7], valor[8], valor[9], valor[10], valor[11],
            valor[12], valor[13], valor[14], valor[15], valor[16]
        )
        csv_filename = os.path.join(MYCSVDIR, nome_csv)

        all_data = []

        for i in range(1, 21):
            frame_atual = frames[i]
            rf_values = frame_atual.fieldOutputs['RF'].values
            le_values = frame_atual.fieldOutputs['LE'].values
            
            # Sum reaction forces on the pulling grip
            rf = sum([abs(v.data[1]) for v in rf_values if v.nodeLabel in nodes_puxa_labels])
            row = [rf]
            
            # Extract strains for each specific element set
            for k in ['Esq', 'Dir', 'Centro']:
                elems_labels = elems_por_set[k]
                num_elems = len(elems_labels)
                
                soma_le11, soma_le22, soma_le12 = 0.0, 0.0, 0.0
                
                for val in le_values:
                    if val.elementLabel in elems_labels:
                        soma_le11 += val.data[0]; soma_le22 += val.data[1]; soma_le12 += val.data[3]
                
                # Append averaged values if the set is not empty
                if num_elems > 0:
                    row.extend([
                        soma_le11/num_elems, soma_le22/num_elems, soma_le12/num_elems
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
                    
            all_data.append(row)

        odb.close()
        
        # Write extracted data to CSV
        with open(csv_filename, 'wb') as file:
            writer = csv.writer(file, delimiter=';')
            cabecalho = ['RF']
            for s in ['Esq', 'Dir', 'Centro']:
                for c in ['LE11', 'LE22', 'LE12']:
                    cabecalho.append("{}_{}".format(s, c))
                    
            writer.writerow(cabecalho)
            writer.writerows(all_data)
            
        print(" ✅ PERFECT CSV! 20 rows saved.")
        timer_tick(start_time)
        limpar_lixo(job_name)

    except Exception as e:
        print(" ❌ Error during extraction: {}".format(e))
        try: odb.close() 
        except: pass
        limpar_lixo(job_name)
        timer_tick(start_time)
        continue
