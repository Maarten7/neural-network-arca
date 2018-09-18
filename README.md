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
1. Get monte carlo (.evt) files from Lyon. (a)nu- eCC eNC for showers (a)numuCC for tracks. Perhaps also tauon and atmospheric muons.
2. Run detector simulation with JPP (example: JTriggerEfficientcy or JEventTimesliceWriter) get .root files. Also make events with only K40 (with JRandomTimeSliceWriter)
3. Transform events from .root files into numpy arrays according to your Neural Network model. Save as .hdf5 files. Also save meta data like Energy, direction, position and type of neutrino.
4. Feed numpy arrays to Neural network model for training. Keep 20% test set. Save weights
5. Feed test set to Neural network save output to .hdf5 to train as trigger, classifier or reconstructor
6. Show results with a lot of plots 
7. Profit
