from keras.models import Sequential

from keras.layers import Conv3D
from keras.layers.normalization import BatchNormalization

from temporal_model import batches

model = Sequential()

model.add(Conv3D(
    input_shape=(None, 400, 13, 13, 18, 3),
    filters=40,
    kernel_size=(3,3,4)))

model.add(BatchNormalization())

model.compile(loss='crossentropy', optimizer='adam')

for events, labels in model.batches(15):
    model.fit(events, labels)

