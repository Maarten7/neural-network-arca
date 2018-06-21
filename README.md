# neural-network-arca
master project Maarten. 

software needed:
    ROOT
    AANET
    JPP

python needed:
    Tensorflow GPU
    h5py
    PyRoot

workflow:
    Get bunch of .evt files from LYON
    Convert them to .root files with JPP trigger efficienty with or without trigger
    Define neural network model in /models. 
    Convert events in .root files into h5py numpy events (data_writer.py)
    Train model (trainer.py)
    Test model (netwerk_tester.py)
    Make plots (plotter.py)
