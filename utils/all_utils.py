# All the extra functions such as preparing utilities and all
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib     # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os

plt.style.use("fivethirtyeight")

def prepare_data(df):
  """It is used to seperate the dependent vriables and independent features

  Args:
      df (pd.DataFrame): Its the pandas DataFrame

  Returns:
      tuple: It returns the tuples of dependent variables and independent variables
  """
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

def save_model(model, filename):
  """This saves the trained model to

  Args:
      model (python object): trained model to
      filename (str): path to save the trained model
  """
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN'T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename (slash based on your OS)
  joblib.dump(model, filePath)  

def save_plot(df, file_name, model):
  def _create_base_plot(df): # create protected function
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle='--', linewidth=1)
    plt.axvline(x=0, color="black", linestyle='--', linewidth=1)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):  # resolution is specified as 0.02
    colors = ("red", "blue", "lightgreen", "gray", "cyan")       # will select colors for unique y values, max 5
    cmap = ListedColormap(colors[: len(np.unique(y))])           # here there are 2 y values,therefore red, blue
    
    X = X.values # as a array. As we want values in array. In above they are in df
    x1 = X[:, 0]  # All rows of X, and column 0
    x2 = X[:, 1]  # All rows of X, and column 1
    x1_min, x1_max = x1.min() -1, x1.max() + 1 # Finding min & max and -1, +1 to get extended boundary for classification
    x2_min, x2_max = x2.min() -1, x2.max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))   #from xx1 min max, find all points at resolution (0.02)
    print(xx1)
    print(xx1.ravel())  # ravel: Make into a single array)
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # WHy transpose..check
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)  # alpha=0.2..transparency, cmap defined above, xx1, xx are found, then based on Z value 0 red, 1 blue (cmap gives value)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()
  
  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOESN'T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)