import numpy as np
from numpy import linalg as LA # import linear algebra 
from numpy.linalg import inv
import sys

# is_max = True
# A1 = []
# b1 = []
# constraint_type = []
# C1 = []
# file = open("input.txt", "r")
# content=file.readlines()
# print(content)
# file.close()

A = [[1,2,3],[4,5,6],[7,8,9]]
b = [[-10],[11],[12]]
c = [2 ,4 ,6]
constraint_type = [1,-1,0]



def simplex_iteration(A, b, C, m: int, n: int):
    #intialization
    Iteration=0
    Z=0
    X=np.zeros((n+m))
    XB=np.zeros((m))
    CB=np.zeros((m))
    XN=np.zeros((n))
    CN=np.zeros((n))
    RC = np.zeros((n+m))
    Basis:int=np.zeros((m))
    B = np.zeros((m,m))
    NB = np.zeros((m,n))
    Index_Enter=-1
    Index_Leave=-1
    eps = 1e-12

    for i in range(0,m):
        Basis[i]=n+i
        for j in range(0,m):
         B[i, j]=A[i,n+j]
        for j in range(0,n):
         NB[i, j]=A[i,j]

    for i in range(0,n):
        CN[i]=C[i]
        print("CN: ", CN[i]) 
  
    RC=C-np.dot(CB.transpose(),np.dot(inv(B),A))
    MaxRC=0
    for i in range(0,n+m):
        if(MaxRC<RC[i]):
         MaxRC=RC[i]
         Index_Enter=i

    print("Basis", Basis)
    C_initial = C.copy()
    C_initial = np.insert(C_initial,0,0)
    Down_Table_initial = np.concatenate((b,A),1)
    Down_Table_initial = np.dot(inv(B),Down_Table_initial)
    Initial_Tableau = np.concatenate((C_initial.reshape(1,-1),Down_Table_initial),0)
    print("----------------Initial_Tableau----------------")
    print(np.round(Initial_Tableau,2))
    print("-----------------------------------------------")

    while(MaxRC > eps):
      Iteration=Iteration+1
      print("=> Iteration: ",Iteration)

      print(" Index_Enter: ",  Index_Enter)
      Index_Leave=-1
      MinVal=1000000
      print("Enter B: ",B)
      for i in range(0,m):
       if(np.dot(inv(B),A)[i,  Index_Enter] > 0):
         bratio=np.dot(inv(B),b)[i]/np.dot(inv(B),A)[i,  Index_Enter]
         print("  bratio: ", bratio)
         if(MinVal > bratio ):
           Index_Leave=i
           print("  Index_Leave: ",Index_Leave)
           MinVal=bratio
           print("  MinVal: ", MinVal)

      if (Index_Leave == -1):
        C_final = RC.copy()
        C_final = np.insert(C_final,0,-Z[0])
        Down_Table_final = np.concatenate((b,A),1)
        Down_Table_final = np.dot(inv(B),Down_Table_final)
        Final_Tableau = np.concatenate((C_final.reshape(1,-1),Down_Table_final),0)
        print("----------------Final_Tableau----------------")
        print(np.round(Final_Tableau,2))
        print("-----------------------------------------------")
        print("unbounded.")
        return Z,X,RC
      
      Basis[Index_Leave]=Index_Enter 
      print("before updated Basis", Basis)
      print("  Index_Leave: ",Index_Leave)
      for i in range(m-1,0,-1):
        if(Basis[i] < Basis[i-1]):
            temp=Basis[i-1]
            Basis[i-1]=Basis[i]
            Basis[i]=temp

      print("updated Basis", Basis)

      for i in range(0,m):
          for j in range(0,n+m):
              if(j==Basis[i]):
                B[:, i]=A[:,j]
                CB[i]=C[j]

      print("Exit Basis", Basis)
      print("Exit B: ",B)

      RC=C-np.dot(CB.transpose(),np.dot(inv(B),A))
      MaxRC=0
      for i in range(0,n+m):
        if(MaxRC<RC[i]):
         MaxRC=RC[i]
         Index_Enter=i
      print("MaxRC",MaxRC)
      X=np.dot(inv(B),b)
      Z=np.dot(CB,X)
    C_final = RC.copy()
    C_final = np.insert(C_final,0,-Z[0])
    Down_Table_final = np.concatenate((b,A),1)
    Down_Table_final = np.dot(inv(B),Down_Table_final)
    Final_Tableau = np.concatenate((C_final.reshape(1,-1),Down_Table_final),0)
    print("----------------Final_Tableau----------------")
    print(np.round(Final_Tableau,2))
    print("-----------------------------------------------")
    print("optimal")
    return Z, X, RC
    
def toStandard(A,b,c,constraint_type):
    A2 = []
    b2 = []
    c2 = []

    s=[]
    ns = []

    for i in range(len(A)):
        if constraint_type[i] != 0:
            s.append(i)
        else:
            ns.append(i)
        
  
    for i in s:
       A2.append(A[i])
    for i in ns:
       A2.append(A[i])

    for i in range (len(s)):
       b2.append([b[s[i]][0]])
       for j in range(len(s)):
          if i == j:
             if constraint_type[s[i]] == -1:
                A2[i].append(-1)
             else:
                A2[i].append(1)
          else:
             A2[i].append(0)
    
    for i in range(len(s),len(A)):
       for j in range(len(s)):
          A2[i].append(0)

    for i in ns:
       b2.append([b[i][0]])

    c2 = c.copy()
    for i in range(len(s)):
       c2.append(0)
    
    for i in range(len(A2)):
       if (b2[i][0]<0):
        b2[i][0] *= -1
        A2[i] = [x * -1 for x in A2[i]]

    row = len(A2)

    for i in range(row):
       c2.append(0)
       for j in range(row):
          if i==j:
             A2[i].append(1)
          else:
             A2[i].append(0)
    
    print("A2",A2)
    print("b2",b2)
    print("c2",c2)

    return A2,b2,c2 
    
       
    
toStandard(A,b,c,constraint_type)    


def get_ans(basis,X,n):
    ans = [0]*n
    for i in range(len(X)):
       if (basis[i]<n):
          ans[basis[i]] = X[i]
    return ans 
       

   

# Example4:

# C=np.array([2,4,6,0,0,0,0,0])
# A=np.array([[1,2,3,1,0,1,0,0],[4,5,6,0,-1,0,1,0],[7,8,9,0,0,0,0,1]])
# b=np.array([[10],[11],[12]])

# Z,X,RC=simplex_iteration(A,b,C,3,5)

# print("Z", Z)
# print("X",X)
# print("RC",RC)


# Example1:
#A=np.array([[1,1,1,3,1,1,0,0],[1,4,1,3,1,0,1,0],[1,2,1,4,1,0,0,1]])
#b=np.array([[1],[2],[3]])
#C=np.array([5,3,4,2,3,0,0,0])


#Example2: 
#A=np.array([[1,3,2,1,0,0],[2,2,1,0,1,0],[1,1,2,0,0,1]])
#b=np.array([[4],[2],[3]])
#C=np.array([2,3,2,0,0,0])

# Example3:
#A=np.array([[1,2,1,2,1,3,1,1,0,0],[1,3,4,2,1,3,1,0,1,0],[3,2,2,1,5,4,1,0,0,1]])
#b=np.array([[4],[4],[4]])
#C=np.array([4,3,5,3,3,4,3,0,0,0])

# Example4:
#A=np.array([[1, 2, 2, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#[3, 5, 1, 4, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#[4, 3, 2, 7, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#[2, 1, 7, 2, 6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#[3, 2, 1, 4, 3, 7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#[1, 2, 5, 2, 5, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#[3, 2, 1, 4, 2, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#[4, 3, 2, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#[2, 1, 4, 2, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#[4, 2, 1, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
#b=np.array([[9],[4],[5],[8],[7],[9],[6],[4],[3],[4]])
#C=np.array([5, 4, 3, 5, 8, 4,0,0,0,0,0,0,0,0,0,0])