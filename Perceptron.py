import numpy as np
import  random
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self,dataset_F,N_Of_epochs,L_R,Classes,Features,Bias):
        self.dataset_F = dataset_F
        self.N_Of_epochs = N_Of_epochs
        self.L_R = L_R
        self.Classes = Classes
        self.Features = Features
        self.Bias = Bias
        self.weights = np.random.rand(1,3)

    def shuffle(self):
      if self.Classes=="C1&C2":
        random.shuffle(self.dataset_F[:50])
        train = np.concatenate((self.dataset_F [:30, :], self.dataset_F[50:80, :]),axis=0)
        random.shuffle(self.dataset_F[50:100])
        test = np.concatenate((self.dataset_F[30:50, :] , self.dataset_F[80:100, :]),axis=0)

      elif self.Classes=="C1&C3":
        random.shuffle(self.dataset_F[:50])
        train = np.concatenate((self.dataset_F [:30, :], self.dataset_F[100:130, :]),axis=0)
        random.shuffle(self.dataset_F[100:150])
        test = np.concatenate((self.dataset_F[30:50, :] , self.dataset_F[130:150, :]),axis=0)

      else:
        random.shuffle(self.dataset_F[50:100])
        train = np.concatenate((self.dataset_F[50:80, :] , self.dataset_F[100:130, :]),axis=0)
        random.shuffle(self.dataset_F[100:150])
        test = np.concatenate((self.dataset_F[80:100, :] , self.dataset_F[130:150, :]),axis=0)

      return train,test

    def signum (self,W,X):
        y = np.dot(X,np.transpose(W))
        if y[0,0]>0:
            return 1
        else:
            return -1

    def calculate_loss(self,t,y):
         return t-y

    def Training_Phase(self,train,Columns1,Columns2):
      if(self.Bias)==1:
          b=np.ones([1,1])
      else:
          b=np.zeros([1,1])

      for i in range(self.N_Of_epochs):
          for j in range(len(train)):
              y = self.signum(self.weights,np.concatenate((b,train[j:j+1,[Columns1,Columns2]]),axis=1))

              if j<30 :
                  t = 1
              else:
                  t = -1
              if y != t:
               self.weights = self.weights+(self.L_R*self.calculate_loss(t,y)*np.concatenate((b,train[j:j+1,[Columns1,Columns2]]),axis=1))




    def draw_line(self,test,Columns1,Columns2):
    #get_points
        b = self.weights[0,0]
        w1 = self.weights[0,1]
        w2 = self.weights[0,2]
        p1=(-1*b)/w1
        p2=((-1*b)-(w1*6))/w2

     #draw_line
        plt.figure("XTest_YTest")
        plt.scatter(test[:20, Columns1], test[:20, Columns2])
        plt.scatter(test[20:40, Columns1], test[20:40, Columns2])
        point1 = [p1, 0]
        point2 = [6, p2]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values)
        plt.xlabel("XTest")
        plt.ylabel("Ytest")
        plt.show()

    def Chosen_Features(self):
        if self.Features=="X1_X2":
            Columns1=0
            Columns2=1
        elif self.Features=="X1_X3":
            Columns1=0
            Columns2=2
        elif self.Features=="X1_X4":
            Columns1=0
            Columns2=3
        elif self.Features=="X2_X3":
            Columns1=1
            Columns2=2
        elif self.Features=="X2_X4":
            Columns1=1
            Columns2=3
        elif self.Features=="X3_X4":
            Columns1=2
            Columns2=3
        return Columns1,Columns2

    def Testing_Phase(self,test, Columns1, Columns2):
        C1_C1=0
        C1_C2=0
        C2_C1=0
        C2_C2=0
        if (self.Bias) == 1:
            b = np.ones([1, 1])
        else:
            b = np.zeros([1, 1])
        print(len(test))
        for j in range(len(test)):
            y = self.signum(self.weights, np.concatenate((b, test[j:j + 1, [Columns1,Columns2]]), axis=1))
            if j < 20:
              if y>0:
                 C1_C1+=1
              elif y<0:
                 C1_C2+=1
            else:
              if y>0:
                C2_C1+=1
              elif y<0:
                C2_C2+=1

        return np.array([[C1_C1,C1_C2],[C2_C1,C2_C2]])


    def classify(self):
        train,test = self.shuffle()
        Columns1, Columns2 = self.Chosen_Features()
        self.Training_Phase(train,Columns1,Columns2)
        Confusion_Matrix=self.Testing_Phase(test,Columns1,Columns2)
        self.draw_line(test,Columns1,Columns2)
        print("Confusion_Matrix",Confusion_Matrix)
        print("accuracy is = ", (Confusion_Matrix[0][0] + Confusion_Matrix[1][1]) / 40)