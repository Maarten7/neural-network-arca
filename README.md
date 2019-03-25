# neural-network-arca
Master project Maarten. 

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
2. Run detector simulation with JPP (example: JTriggerEfficientcy or JEventTimesliceWriter) get .root files. Also make events with only K40 (with JRandomTimeSliceWriter). Here use the custom clock that makes time-blocks (20000 ns) instead of timeslices (100 miliseconds).
3. Transform events from .root files into numpy arrays according to your Neural Network model. (models/) Save as .hdf5 files. Also save meta data like Energy, direction, position and type of neutrino. (network_data_writer.py)
4. Feed numpy arrays to Neural network model to train as trigger, classifier or reconstructor. Keep 20% test set. Save weights (network_trainer.py)
5. Feed test set to Neural network save output to .hdf5  (network_tester.py)
6. Show results with a lot of plots (network_plotter.py)
7. Profit

#### The model
1. A image is made from the DOM line id's to a matrix x and y index. 
2. DOM z positions index is used for the matrix z index.
3. For a given time interval all the hits on a dom (one matrix element) is added together.
4. Here either the TOT or the Number of hits are added.
5. This value can also be weighted with the direction vector of the PMT that is hit. Than each matrix element becomes a 3 vector.
6. All non DOM or non-hit matrix element are 0.
7. Neural network like data that is mostely in range between -1 and 1. So the matrix elements should/can be normelized.
##### Time independent model DEPRECATED
1. The time interval is the duration of a whole event (~12000 ns) and all the hits on a dom are added together.
2. You now have a (13, 13, 18, 3) or (13, 13, 18, 1) shaped matrix where the TOTs or NumHits are weighted with de PMT direction or not respectively. (x, y, z, c)
3. This matrix/numpy array goed into a 3D convolutional network where the RGB channel can be used for the weighted 3 vector.
4. Number of layers and nodes can vary. But here 3 3dConv layers and 2 fully connected layers are used. With .. .. .. nodes.
##### Time dependent model
1. The time interval can be any length but a length of 20000 ns is chosen.
2. For each 400 ns (~lighttime betweens 2 doms) all hits are added as in the TID model. I call this a minitimeslice
3. You now have a (50, 13, 13, 18, 3) or (50, 13, 13, 18, 1) shaped matrix where the TOTs or NumHits are weighted with de PMT direction or not respectively. (t, x, y, z, c)
4. Each minitimeslice matrix with shape (13, 13, 18, 3) is passed through a 3Dconv similar to the TID model.
5. The outputs of each minitimeslice are passed in order through a LSTM cell for reaching the final output
#### Training
Cost funtion:   (softmax) Cross entropy
Optimizer:      Adam optimizer with learning rate = 0.003 * .97 ^ (num epoch)

#### Trouble shooting
If you get a CUDA_ERROR on schol or schar ask Jan Just Keijser or Roel Aaij to reset it.
