import numpy as np
import neurolab as nl

# Букви Н, А, В
target = [
    [-1,0,0,0, -1,-1,0,0,0, -1,-1,-1,-1,0,-1,-1, 0,0,0, -1,-1,0,0,-1,-1],  # Н
    [0,-1,0,-1,0,-1,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,-1,-1,0,0,0,-1],  # А
    [-1,-1,-1,-1,0,-1,0,-1,0,-1,-1,-1,0,-1,0,-1,0,0,0,-1,-1,0,-1,-1,0]   # В
]

chars = ['Н', 'А', 'В']
target = np.asfarray(target)
target[target == 0] = -1

# Створення та тренування мережі
net = nl.net.newhop(target)
output = net.sim(target)
print("Тест на навчальних зразках:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

# Тестування букви Н з помилками
print("\nТест на зіпсованій букві Н:")
test = np.asfarray([1,0,0,0,1, 1,0,1,0,1, 1,1,1,1,1, 1,0,1,0,1, 1,0,0,0,1])  # Змінено кілька пікселів
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Кількість кроків симуляції', len(net.layers[0].outs))
