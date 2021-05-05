import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from Perceptron import *
from Adaline import *

from PIL import ImageTk,Image


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


# print(Dataset_Of_Features)
# print(Dataset_Of_Features[:3,:3:2])

top = Tk()
top.geometry('500x500')
top.config(bg='BLUE')
# conv=Canvas(top,width=500,height=1200)
# img=ImageTk.PhotoImage(Image.open('bc.jpg'))
# conv.create_image(0,0,anchor=NW,image=img)
# conv.pack()

def collect_data():

   Feature = variable_Feature.get()
   Classes = variable_classes.get()
   Number_Of_Epochs = Eepochs.get()
   Learing_rate = Erate.get()
   Bias = Cbias.get()
   threshold=Threshold_txt.get()
   return Feature,Classes,Number_Of_Epochs,Learing_rate,Bias,threshold
   #print(float(threshold), float(Learing_rate), Classes, Feature, Bias)

def run_Perceptron():
    Feature, Classes, Number_Of_Epochs, Learing_rate, Bias, threshold=collect_data()
    p = Perceptron(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature, Bias)
    p.classify()
def run_Adaline():
   Feature, Classes, Number_Of_Epochs, Learing_rate, Bias, threshold = collect_data()
   adaline =Adaline(Dataset_Of_Features,int( Number_Of_Epochs), float(Learing_rate), Classes, Feature,float(threshold ),Bias)
   adaline.classify()

def run_backprobagation():
    pass


Cbias = IntVar()
variable_Feature = StringVar(top)
variable_classes = StringVar(top)
variable_Lepochs = StringVar(top)
variable_Lrate = StringVar(top)
variable_threshold= StringVar(top)

variable_Feature.set("X1_X2")
variable_classes.set("C1&C2")
variable_Lepochs.set("Enter number of epochs")
variable_Lrate.set("Enter learning rate")
variable_threshold.set("Enter threshold")

MI_Feature = OptionMenu(top,variable_Feature,"X1_X2","X1_X3","X1_X4","X2_X3","X2_X4","X3_X4")
MI_Feature.config(bg='Pink')
MI_classes = OptionMenu(top,variable_classes,"C1&C2","C1&C3","C2&C3")
MI_classes.config(bg='Pink')

Lepochs = Label(top,textvariable=variable_Lepochs,bg='BLUE',fg='PINK',font='Andalus 10 italic bold')
Lrate = Label(top,textvariable=variable_Lrate,bg='BLUE',fg='PINK',font='Andalus 10 italic bold')
Lthreshold=Label(top,textvariable=variable_threshold,bg='BLUE',fg='PINK',font='Andalus 10 italic bold')

Eepochs = Entry(top)
Erate = Entry(top)
Threshold_txt=Entry(top)
Checkbias = Checkbutton(top,bg='BLUE',fg='PINK', text="Bias",font='bold',variable=Cbias)
Perceptron_btn = Button(top,bg='pink', text="classifiy using perceptron", width=20, command = run_Perceptron)
Adaline_btn = Button(top, bg='pink',text="classifiy using Adaline", width=20, command = run_Adaline)


MI_Feature.place(x=20,y=10)
MI_classes.place(x=20,y=50)
Lepochs.place(x=20,y=90)
Eepochs.place(x=190,y=90)
Lrate.place(x=20,y=130)
Lthreshold.place(x=20,y=170)

Erate.place(x=160,y=130)
Threshold_txt.place(x=160,y=170)
Checkbias.place(x=20,y=220)
Perceptron_btn.place(x=100, y=200)
Adaline_btn.place (x=300, y=200)
mainloop()




#Function_collectdata that is the action pf button, the start of program,#####line_41




