
import numpy as np

from ann_visualizer.visualize import ann_viz
from graphviz import Source
from keras.models import Sequential
from keras.layers import Dense

training_data=np.array([[0,0], [0,1],[1,0],[1,1]], "float32")
target_data=np.array([[0], [1], [1], [0]], "float32")

model=Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_square_error', optimizer='adam', metrics=['accuracy'])
ann_viz(model, title='Simple XOR Classifier')
graph_source=Source.from_file('network.gv')
graph_source
graph_source.source
print(graph_source.source)
model.get_weights()
