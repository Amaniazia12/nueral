import numpy as np
import  random
import matplotlib.pyplot as plt
import math
class Adaline:
    def __init__(self,dataset_F,N_Of_epochs,L_R,Classes,Features,Threshold,Bais):
        self.dataset_F = dataset_F
        self.N_Of_epochs = N_Of_epochs
        self.L_R = L_R
        self.Classes = Classes
        self.Features = Features
        self.Threshold=Threshold
        self.Bias = Bais
        self.weights = np.random.rand(1,3)

    def shuffle(self):
      if self.Classes=="C1&C2":
        random.shuffle(self.dataset_F[:50])
        random.shuffle(self.dataset_F[50:100])
        train = np.concatenate((self.dataset_F [:30, :], self.dataset_F[50:80, :]),axis=0)
        test = np.concatenate((self.dataset_F[30:50, :] , self.dataset_F[80:100, :]),axis=0)

      elif self.Classes=="C1&C3":
        random.shuffle(self.dataset_F[:50])
        random.shuffle(self.dataset_F[100:150])
        train = np.concatenate((self.dataset_F [:30, :], self.dataset_F[100:130, :]),axis=0)
        test = np.concatenate((self.dataset_F[30:50, :] , self.dataset_F[130:150, :]),axis=0)

      else:
        random.shuffle(self.dataset_F[50:100])
        random.shuffle(self.dataset_F[100:150])
        train = np.concatenate((self.dataset_F[50:80, :] , self.dataset_F[100:130, :]),axis=0)
        test = np.concatenate((self.dataset_F[80:100, :] , self.dataset_F[130:150, :]),axis=0)

      return train,test

    def signum (self,W,X):
        y = np.dot(X,np.transpose(W))
        if y[0,0]>0:
            return 1
        elif y[0,0]== 0 :
            return 0
        else :
            return -1

    def calculate_loss(self,t,y):
         return t-y

    def Training_Phase(self,train,Columns1,Columns2):
        MSE=0
        commulativeError=0
        if (self.Bias) == 1:
             b = np.ones([1, 1])
        else:
             b = np.zeros([1, 1])
        while (MSE >=self.Threshold):

         for i in range(self.N_Of_epochs):
          for j in range(len(train)):

              y = np.dot(np.concatenate((b,train[j:j+1,[Columns1,Columns2]]),axis=1), np.transpose(self.weights ))

              if j<30 :
                  t = 1
              else:
                  t = -1
              error = self.calculate_loss(y, t)
              if y != t:
               self.weights = self.weights+(self.L_R*error*np.concatenate((train[j:j+1,[Columns1,Columns2]]),axis=1))
          for i in range(len(train)):
              y = np.dot(train[j:j + 1, [Columns1, Columns2]], np.transpose(self.weights))
              if j<30 :
                  t = 1
              else:
                  t = -1
              if y != t:
               commulativeError+=math.sqrt(self.calculate_loss(y,t))
          MSE=commulativeError/len(train)

    def draw_line(self, train, test, Columns1, Columns2):
        # get_points
        b = self.weights[0, 0]
        w1 = self.weights[0, 1]
        w2 = self.weights[0, 2]

        min_x1 = min(train[:, Columns1]) - 1
        max_x1 = max(train[:, Columns1]) + 1
        x1_values = [min_x1, max_x1]
        x2_values = (-1 * b - np.multiply(w1, x1_values)) / w2

        # draw_line between training data
        plt.figure("XTtain_YTrain")
        plt.scatter(train[:30, Columns1], train[:30, Columns2])
        plt.scatter(train[30:60, Columns1], train[30:60, Columns2])
        plt.plot(x1_values, x2_values)
        plt.xlabel("X1Train")
        plt.ylabel("x2Train")
        plt.show()

        plt.figure("XTest_YTest")
        plt.scatter(test[:20, Columns1], test[:20, Columns2])
        plt.scatter(test[20:40, Columns1], test[20:40, Columns2])
        plt.plot(x1_values, x2_values)
        plt.xlabel("X1Test")
        plt.ylabel("x2Test")
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
        for j in range(len(test)):
            print(np.shape(self.weights))

            y = self.signum(self.weights,np.concatenate((b, test[j:j + 1, [Columns1,Columns2]]), axis=1))

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
        self.draw_line(train,test,Columns1,Columns2)
        print("Confusion_Matrix",Confusion_Matrix)
        print("accuracy is = ", (Confusion_Matrix[0][0] + Confusion_Matrix[1][1]) / len(test))