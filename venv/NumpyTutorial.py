# importação da biblioteca numpy
import numpy as np

# criação de matriz de zeros com 2 linhas e 3 colunas
x = np.zeros((2, 3))
print("x = \n" + str(x))

# criação se matriz de zeros com 4 linhas e 5 colunas
y = np.zeros((4, 5))
print("\ny = \n" + str(y))

# criação de um array com 10 elementos todos eles 0
z = np.zeros((1, 10))
print("\nz = \n" + str(z))

# criação de uma matriz 3x2 com os elementos que queremos
matriz1 = np.array([[1, 2],
                    [0, 3.2],
                    [1, 7]])
print("\nmatriz1 = \n" + str(matriz1))

#Verificar as dimensões das variaveis/objetos anteriormente criados
print("\nDimensão da matriz x:" + str(x.shape) +
      "\nNúmero de linhas:" + str(x.shape[0]) +
      "\nNúmero de colunas:" + str(x.shape[1]))

print("\nDimensão da matriz y:" + str(y.shape) +
      "\nNúmero de linhas:" + str(y.shape[0]) +
      "\nNúmero de colunas:" + str(y.shape[1]))
#Verificar que está como sendo mattriz de 1x10
print("\nDimensão do array z:" + str(z.shape) +
      "\nNúmero de linhas:" + str(z.shape[0]) +
      "\nNúmero de colunas:" + str(z.shape[1]))

print("\nDimensão da matriz1:" + str(matriz1.shape) +
      "\nNúmero de linhas:" + str(matriz1.shape[0]) +
      "\nNúmero de colunas:" + str(matriz1.shape[1]))

# Criação de array com os numeros entre 10 e 30 com passo igual a 5, sendo que o 30 já não entra
a = np.arange(10, 30, 5)
print("\na = \n" + str(a))

#Neste o 30 já entra
b = np.arange(10, 31, 5)
print("\nb = \n" + str(b))

#Array composto por 9 elementos entre o 0 e o 2
c = np.linspace(0, 2, 9)
print("\nc = \n" + str(c))