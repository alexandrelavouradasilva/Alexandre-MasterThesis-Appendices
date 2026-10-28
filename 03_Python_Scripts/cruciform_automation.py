"""
Automation of Cruciform Test Simulations

The Python script used in this work adapts the framework developed by Marques [https://ria.ua.pt/handle/10773/44118]. 
This script automates the simulation of the CPB’06 yield criterion and Swift hardening law 
using Abaqus on the cruciform test. It defines a 3D model, sets up material properties, 
and creates a mesh for a cruciform geometry. The script calls a pre-calculated Design 
of Experiments (DOE) with varying material parameters from an external CSV file, executes 
the simulation, and extracts results such as displacements and strains from the Output 
Database (ODB). Finally, it saves the processed data into CSV files for further analysis.
"""

# -*- coding: mbcs -*-
# Do not delete the following import lines
# abaqus cae noGUI=Cruciform.py
# This is the 3D Cruciform NM100 Mesh
from abaqus import *
from abaqusConstants import *
from array import array
from subprocess import call

import __main__
import glob
import os
import csv
import time
import sys
import numpy as np
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import random

# ========================================================================= #
# HELPER FUNCTIONS
# ========================================================================= #

def make_dir(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def timer_tick(start_time):
    """Calculates and prints the elapsed time."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    print("Finished in {}:{:02d} minutes.".format(elapsed_minutes, elapsed_seconds))

def run_job():
    """Executes the Abaqus job via the command line."""
    print("Running the job ...")
    try:
        retcode = call("abaqus job=Job_Alexandre user=UMMDp_FLC.f interactive ask_delete=OFF", shell=True)
        if retcode < 0:
            print >>sys.stderr, "Job was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Job returned", retcode
    except OSError as e:
        print >>sys.stderr, "Job execution failed:", e


# ========================================================================= #
# FOLDERS AND PATHS
# ========================================================================= #
current_dir = os.getcwd()
MYCSVDIR = os.path.join(current_dir, 'data', 'ofat')
MYCSVDIR = os.path.normpath(MYCSVDIR)
make_dir(MYCSVDIR)

# Path to the master Design of Experiments (DOE) file
caminho_doe_mestre = os.path.join(current_dir, 'data', 'doe', 'DOE.csv')

#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------  Define the Extrusion Part and Partition  -----------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)

# Sketching the 2D profile of the cruciform specimen (one eighth due to symmetry)
s.Line(point1=(0.0, 0.0), point2=(30.0, 0.0))
s.HorizontalConstraint(entity=g[2], addUndoState=False)
s.Line(point1=(30.0, 0.0), point2=(30.0, 15.0))
s.VerticalConstraint(entity=g[3], addUndoState=False)
s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
s.Line(point1=(0.0, 0.0), point2=(0.0, 30.0))
s.VerticalConstraint(entity=g[4], addUndoState=False)
s.PerpendicularConstraint(entity1=g[2], entity2=g[4], addUndoState=False)
s.Line(point1=(0.0, 30.0), point2=(15.0, 30.0))
s.HorizontalConstraint(entity=g[5], addUndoState=False)
s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
s.FixedConstraint(entity=v[0])
s.EqualLengthConstraint(entity1=g[2], entity2=g[4])
s.EqualLengthConstraint(entity1=g[3], entity2=g[5])

# Creating the central notch
s.CircleByCenterPerimeter(center=(15.0, 15.0), point1=(22.5, 23.75))
s.Line(point1=(15.0, 30.0), point2=(15.0, 26.5244305715896))
s.VerticalConstraint(entity=g[7], addUndoState=False)
s.PerpendicularConstraint(entity1=g[5], entity2=g[7], addUndoState=False)
s.CoincidentConstraint(entity1=v[7], entity2=g[6], addUndoState=False)
s.Line(point1=(30.0, 15.0), point2=(26.5244305715896, 15.0))
s.HorizontalConstraint(entity=g[8], addUndoState=False)
s.PerpendicularConstraint(entity1=g[3], entity2=g[8], addUndoState=False)
s.CoincidentConstraint(entity1=v[8], entity2=g[6], addUndoState=False)
s.autoTrimCurve(curve1=g[6], point1=(19.2347831726074, 25.6309909820557))
s.autoTrimCurve(curve1=g[9], point1=(24.0811347961426, 21.9118900299072))
s.FixedConstraint(entity=v[11])
s.RadialDimension(curve=g[10], textPoint=(3.96877670288086, 3.67629909515381), radius=7.0)

# Fillets
s.FilletByRadius(radius=2.5, curve1=g[7], nearPoint1=(14.8730659484863, 23.1115970611572), curve2=g[10], nearPoint2=(13.4191551208496, 21.7919178009033))
s.FilletByRadius(radius=2.5, curve1=g[8], nearPoint1=(22.9907035827637, 14.9535703659058), curve2=g[10], nearPoint2=(22.1425971984863, 13.7538595199585))
s.ObliqueDimension(vertex1=v[3], vertex2=v[4], textPoint=(8.93628692626953, 38.5878562927246), value=15.0)

# Adjusting sketch entities
s.dragEntity(entity=v[14], points=((12.5, 24.1651513899117), (12.5, 23.75), (13.75, 26.25), (13.75, 27.5)))
s.dragEntity(entity=v[4], points=((15.0, 30.0), (15.0, 30.0), (16.25, 30.0), (13.75, 30.0), (11.25, 28.75)))
s.undo()
s.ObliqueDimension(vertex1=v[0], vertex2=v[1], textPoint=(14.1344108581543, -7.11508655548096), value=30.0)

# Extruding the sketch to create a 3D deformable body
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidExtrude(sketch=s, depth=0.5)
s.unsetPrimaryObject()

p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
p = mdb.models['Model-1'].parts['Part-1']

# Creating a partition on the face
f, e, d1 = p.faces, p.edges, p.datums
t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[19], sketchPlaneSide=SIDE1, origin=(12.169049, 12.169049, 0.5))
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=84.85, gridSpacing=2.12, transform=t)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Model-1'].parts['Part-1']
p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
s1.rectangle(point1=(-12.169049, -12.169049), point2=(-2.11879614980537, -2.11879678680635))
s1.CoincidentConstraint(entity1=v[13], entity2=g[5], addUndoState=False)
s1.EqualDistanceConstraint(entity1=v[3], entity2=v[5], midpoint=v[13], addUndoState=False)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
pickedFaces = f.getSequenceFromMask(mask=('[#200 ]', ), )
e1, d2 = p.edges, p.datums
p.PartitionFaceBySketch(sketchUpEdge=e1[19], faces=pickedFaces, sketch=s1)
s1.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']

# Display settings
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=OFF)


#  ----------------------------------------------------------------------------------------------------------------  #
# -------------------------------------------  Define the Material  -----------------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

# Material definition using User Defined Material (UMAT)
mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Depvar(n=7)
mdb.models['Model-1'].materials['Material-1'].UserOutputVariables(n=15)
mdb.models['Model-1'].materials['Material-1'].UserMaterial(mechanicalConstants=())

# Section definition and assignment
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=None)
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
region = p.Set(cells=cells, name='Set-1')
p = mdb.models['Model-1'].parts['Part-1']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# Material orientation
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(cells=cells)
orientation=None
mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_3, localCsys=orientation, fieldName='', additionalRotationType=ROTATION_ANGLE, additionalRotationField='', angle=0.0, stackDirection=STACK_3)

# Assembly setup
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(adaptiveMeshConstraints=ON)

# Define Analysis Step
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', maxNumInc=100000, initialInc=0.1, minInc=1e-20, nlgeom=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')

# Field Output Requests
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=('CDISP', 'CF', 'CSTRESS', 'LE', 'PE', 'PEEQ', 'PEMAG', 'RF', 'S', 'TF', 'U','SDV','UVARM', 'COORD'), numIntervals=20)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)


#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------------  Boundary Conditions Defining  ----------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

# X-direction displacement boundary condition (2mm)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#100 ]', ), )
region = a.Set(faces=faces1, name='Set-1')
mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Step-1', region=region, u1=2.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)

# Y-direction displacement boundary condition (2mm)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#4 ]', ), )
region = a.Set(faces=faces1, name='Set-2')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', region=region, u1=0.0, u2=2.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)

# X-symmetry boundary condition
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#400 ]', ), )
region = a.Set(faces=faces1, name='Set-3')
mdb.models['Model-1'].XsymmBC(name='BC-3', createStepName='Step-1', region=region, localCsys=None)

# Y-symmetry boundary condition
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#200 ]', ), )
region = a.Set(faces=faces1, name='Set-4')
mdb.models['Model-1'].YsymmBC(name='BC-4', createStepName='Step-1', region=region, localCsys=None)

# Z-symmetry boundary condition
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#1000 ]', ), )
region = a.Set(faces=faces1, name='Set-5')
mdb.models['Model-1'].ZsymmBC(name='BC-5', createStepName='Step-1', region=region, localCsys=None)


#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------------------  Mesh Defining  -------------------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

# Mesh setup and generation
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(meshTechnique=ON)

p = mdb.models['Model-1'].parts['Part-1']
p.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)

# Element types: C3D8R (8-node linear brick with reduced integration)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, hourglassControl=ENHANCED, distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)

p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, elemType3))

p = mdb.models['Model-1'].parts['Part-1']
p.generateMesh()

session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(meshTechnique=OFF)


# ---------------------------------------------------------------------------------------------------------------- #
# -------------------------------- READ EXTERNAL DOE AND START SIMULATION LOOP ----------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #

print("Reading the Master DOE Database...")
materiais_virtuais = []

# Read DOE material parameters from CSV
with open(caminho_doe_mestre, 'r') as f_doe:
    leitor = csv.reader(f_doe, delimiter=';')
    primeira_linha = next(leitor)
    
    for linha in leitor:
        if not linha or len(linha) < 17:
            continue
        try:
            materiais_virtuais.append([float(x) for x in linha])
        except ValueError:
            pass 

MATERIAL_INICIAL = 1

# Loop over each material in the DOE
for index, valor in enumerate(materiais_virtuais, start=1):
        
    if index < MATERIAL_INICIAL:
        continue

    # Extract CPB'06 and Swift parameters from the current DOE row
    v_C11, v_C12, v_C13 = valor[0], valor[1], valor[2]
    v_C21, v_C22, v_C23 = valor[3], valor[4], valor[5]
    v_C31, v_C32, v_C33 = valor[6], valor[7], valor[8]
    v_C44, v_C55, v_C66 = valor[9], valor[10], valor[11]
        
    v_a = valor[12]
    v_k2006 = valor[13]
    v_sigma0 = valor[14]
    v_k = valor[15]
    v_n = valor[16]
        
    v_E = 210000.0
    v_poisson = 0.3
    v_e0 = round(float((v_sigma0 / v_k)**(1.0 / v_n)), 14)

    # CORRECTION FOR FORTRAN BUG (SWAP C44 <-> C66)
    umat_C44 = v_C66
    umat_C66 = v_C44
    umat_C55 = v_C55

    start_time = time.time()
    print('\n[CRUCIFORM TEST] Sample {} of {}:'.format(index, len(materiais_virtuais)))
    
    # Update material properties for the user subroutine
    mdb.models['Model-1'].materials['Material-1'].UserMaterial(mechanicalConstants=(
        0, 0, v_E, v_poisson, 3, 
        v_C11, v_C12, v_C13, v_C21, v_C22, v_C23, 
        v_C31, v_C32, v_C33, umat_C44, umat_C55, umat_C66, 
        v_a, v_k2006, 2, v_k, v_e0, v_n, 0
    ))

    job_name = 'Job_Alexandre'
    
    # Delete existing job if it has the same name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    # Create the job without submitting it directly through Python
    mdb.Job(name=job_name, model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, numDomains=4,  
        numGPUs=0)
    
    # Write the .inp file and run via terminal (safer for the Fortran compiler)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    run_job()

    # Preparation for data extraction
    node_set_x = mdb.models['Model-1'].rootAssembly.sets['Set-1']
    node_set_y = mdb.models['Model-1'].rootAssembly.sets['Set-2']
    nodes_x = [node.label for node in node_set_x.nodes]
    nodes_y = [node.label for node in node_set_y.nodes]

    odb_Path = job_name + '.odb'
    
    try:
        # Attempt to open the generated ODB file
        cruci_ODB = session.openOdb(name=odb_Path)
        frames = len(cruci_ODB.steps['Step-1'].frames)
        
        # If simulation diverged early and saved very few frames, skip data extraction
        if frames < 5:
            print("❌ Abaqus diverged early (only {} frames). Skipping...".format(frames))
            cruci_ODB.close()
            timer_tick(start_time)
            continue
        
        # 1. Define destination folder
        MYCSVDIR = r"C:\Users\user\Desktop\Dados"
        if not os.path.exists(MYCSVDIR):
            os.makedirs(MYCSVDIR)
            
        # 2. Save file with 5 decimal places precision
        # (to respect Windows path length limits)
        nome_ficheiro = (
            '{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_'
            '{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_'
            '{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.5f}.csv'
        ).format(
            v_C11, v_C12, v_C13, v_C21, v_C22, v_C23, 
            v_C31, v_C32, v_C33, v_C44, v_C55, v_C66, 
            v_a, v_k2006, v_sigma0, v_k, v_n
        )
            
        csv_filename = os.path.join(MYCSVDIR, nome_ficheiro)
        
        all_data = []

        # 3. Bulletproof extraction loop
        for i in range(1, frames):
            data = []
            somax = 0  
            somay = 0
            
            # Extract total reaction forces (TF)
            tf_field = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['TF'].values
            tf_dict = {v.nodeLabel: v.data for v in tf_field}
            
            for nx in nodes_x:
                if nx in tf_dict:
                    somax += abs(tf_dict[nx][0])
            for ny in nodes_y:
                if ny in tf_dict:
                    somay += abs(tf_dict[ny][1])
            
            data.extend([somax, somay])

            # Extract logarithmic strains (LE)
            le_values = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['LE'].values
            LE_data = np.array([v.data for v in le_values])
            
            # Select relevant strain components (e.g., LE11, LE22, LE12)
            strains = LE_data[:, [0,1,3]].flatten()
            data.extend(["{:.6g}".format(x) for x in strains])

            all_data.append(data)
        
        cruci_ODB.close()

        # Write extracted data to CSV
        with open(csv_filename, 'wb') as file:
            writer = csv.writer(file)
            writer.writerows(all_data)
            
        timer_tick(start_time)
        print("✅ Data successfully extracted. File saved!")

    except Exception as e:
        import traceback
        print("❌ Fatal error during data extraction. Error: {}".format(e))
        traceback.print_exc()
        if 'cruci_ODB' in locals():
            cruci_ODB.close()
        timer_tick(start_time)
        continue
}
