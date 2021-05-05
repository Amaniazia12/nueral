'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from Perceptron import *
from Adaline import *


#loading data
def read_data():
 with open('IrisData.txt') as f:
     lines = f.readlines()
 f.close()
 dataset_X = np.empty([150,4])
 for i in range (len(lines)):
   if i!=0:
     dataset_X[i-1] = lines[i].split(',')[:4]
 return dataset_X



#Plot_data
def plot(X_axis,Y_axis,dataset_X,y1,y2):
 plt.figure(X_axis+'_'+Y_axis)
 plt.scatter(dataset_X[:50,y1],dataset_X[:50,y2])
 plt.scatter(dataset_X[50:100,y1],dataset_X[50:100,y2])
 plt.scatter(dataset_X[100:150, y1], dataset_X[100:150, y2])

 plt.xlabel(X_axis)
 plt.ylabel(Y_axis)
 plt.show()

Dataset_Of_Features = read_data()
plot("X1", "X2", Dataset_Of_Features, 0, 1)
plot("X1", "X3", Dataset_Of_Features, 0, 2)
plot("X1", "X4", Dataset_Of_Features, 0, 3)
plot("X2", "X3", Dataset_Of_Features, 1, 2)
plot("X2", "X4", Dataset_Of_Features, 1, 3)
plot("X3", "X4", Dataset_Of_Features, 2, 3)

top = Tk()
top.geometry('500x500')
def collect_data():

   Feature = variable_Feature.get()
   Classes = variable_classes.get()
   Number_Of_Epochs = Eepochs.get()
   Learing_rate = Erate.get()
   ThresholdVal=EThreshold.get() ###
   Bias = Cbias.get()



   p = Perceptron(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature,ThresholdVal )
   p.classify()
   #adaline =Adaline(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature,float(ThresholdVal))

   #adaline.classify()



Cbias = IntVar()
variable_Feature = StringVar(top)
variable_classes = StringVar(top)
variable_Lepochs = StringVar(top)
variable_Lrate = StringVar(top)
variable_Threshold = StringVar(top)  ##


Cbias = IntVar()
variable_Feature.set("X1_X2")
variable_classes.set("C1&C2")
variable_Lepochs.set("Enter num of epochs")
variable_Lrate.set("Enter learning rate")
variable_Threshold.set("Enter Threshold") ##

MI_Feature = OptionMenu(top,variable_Feature,"X1_X2","X1_X3","X1_X4","X2_X3","X2_X4","X3_X4")
MI_classes = OptionMenu(top,variable_classes,"C1&C2","C1&C3","C2&C3")
Lepochs = Label(top,textvariable=variable_Lepochs)
Lrate = Label(top,textvariable=variable_Lrate)
LThreshold = Label(top,textvariable=variable_Threshold)
Eepochs = Entry(top)
Erate = Entry(top)
EThreshold = Entry(top)  ##


Checkbias = Checkbutton(top, text="Bias",variable=Cbias)
collectdata = Button(top, text="classifiy", width=10, command = collect_data)



MI_Feature.place(x=20,y=10)
MI_classes.place(x=20,y=50)
Lepochs.place(x=20,y=90)
Eepochs.place(x=160,y=90)
Lrate.place(x=20,y=130)
Erate.place(x=130,y=130)
LThreshold.place(x=20,y=170)
EThreshold.place(x=130,y=170)
Checkbias.place(x=20,y=200)
collectdata.place(x=200,y=250)

mainloop()




#Function_collectdata that is the action pf button, the start of program,#####line_41
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from Perceptron import *
from Adaline import *

#loading data
def read_data():
 with open('IrisData.txt') as f:
     lines = f.readlines()
 f.close()
 dataset_X = np.empty([150,4])
 for i in range (len(lines)):
   if i!=0:
     dataset_X[i-1] = lines[i].split(',')[:4]
 return dataset_X



#Plot_data
def plot(X_axis,Y_axis,dataset_X,y1,y2):
 plt.figure(X_axis+'_'+Y_axis)
 plt.scatter(dataset_X[:50,y1],dataset_X[:50,y2])
 plt.scatter(dataset_X[50:100,y1],dataset_X[50:100,y2])
 plt.scatter(dataset_X[100:150, y1], dataset_X[100:150, y2])

 plt.xlabel(X_axis)
 plt.ylabel(Y_axis)
 plt.show()

Dataset_Of_Features = read_data()
plot("X1", "X2", Dataset_Of_Features, 0, 1)
plot("X1", "X3", Dataset_Of_Features, 0, 2)
plot("X1", "X4", Dataset_Of_Features, 0, 3)
plot("X2", "X3", Dataset_Of_Features, 1, 2)
plot("X2", "X4", Dataset_Of_Features, 1, 3)
plot("X3", "X4", Dataset_Of_Features, 2, 3)

top = Tk()
top.geometry('500x500')
def collect_data():

   Feature = variable_Feature.get()
   Classes = variable_classes.get()
   Number_Of_Epochs = Eepochs.get()
   Learing_rate = Erate.get()
   ThresholdVal = EThreshold.get()
   Bias = Cbias.get()



   #p = Perceptron(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature, Bias)
   #p.classify()

   adaline =Adaline(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature,float(ThresholdVal),Bias)
   adaline.classify()


Cbias = IntVar()
variable_Feature = StringVar(top)
variable_classes = StringVar(top)
variable_Lepochs = StringVar(top)
variable_Lrate = StringVar(top)
variable_Threshold = StringVar(top)

variable_Feature.set("X1_X2")
variable_classes.set("C1&C2")
variable_Lepochs.set("Enter number of epochs")
variable_Lrate.set("Enter learning rate")
variable_Threshold.set("Enter Threshold")

MI_Feature = OptionMenu(top,variable_Feature,"X1_X2","X1_X3","X1_X4","X2_X3","X2_X4","X3_X4")
MI_classes = OptionMenu(top,variable_classes,"C1&C2","C1&C3","C2&C3")
Lepochs = Label(top,textvariable=variable_Lepochs)
Lrate = Label(top,textvariable=variable_Lrate)
LThreshold = Label(top,textvariable=variable_Threshold)
Erate = Entry(top)
Eepochs = Entry(top)
EThreshold = Entry(top)
Checkbias = Checkbutton(top, text="Bias",variable=Cbias)
collectdata = Button(top, text="classifiy", width=10, command = collect_data)



MI_Feature.place(x=20,y=10)
MI_classes.place(x=20,y=50)
Lepochs.place(x=20,y=90)
Eepochs.place(x=160,y=90)
Lrate.place(x=20,y=130)
Erate.place(x=130,y=130)
LThreshold.place(x=20,y=170)
EThreshold.place(x=130,y=170)
Checkbias.place(x=20,y=200)
collectdata.place(x=200,y=250)
mainloop()
#Function_collectdata that is the action pf button, the start of program,#####line_41



