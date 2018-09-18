# neural-network-arca
master project Maarten. 

#### Software needed:
    ROOT
    AANET
    JPP

#### Python needed:
    Tensorflow GPU
    h5py
    PyRoot
   
#### Workflow: 
1. Get .evt files from Lyon ( (a)nu eCC eNC muCC )  or also ( tauon and atmospheric muons)
2. Run detector simulation with JPP (example: JTriggerEfficientcy or JEventTimesliceWriter) get .root files. Also make events with only K40 (with JRandomTimeSliceWriter)
3. Transform events from .root files into numpy arrays according to your Neural Network model. Save as .hdf5 files
4. Feed numpy arrays to Neural network model for training. Keep 20% test set. Save weights
5. Feed test set to Neural network save output to .hdf5
6. Plot results
7. Profit
