import neurolab as nl
import numpy as np
import pylab as pl

# Створюємо вхідні дані, використовуючи функцію синуса
i1 = np.sin(np.arange(0, 20))
i2 = np.sin(np.arange(0, 20)) * 2

# Створюємо цільові дані
t1 = np.ones([1, 20])
t2 = np.ones([1, 20]) * 2

# Формуємо масив вхідних даних і цільових значень
input = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
target = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)

# Створюємо нейронну мережу з двома шарами:
# Перший шар з 10 нейронів з активаційною функцією TanSig
# Другий шар з 1 нейроном з активаційною функцією PureLin
net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

# Ініціалізуємо шари випадковими вагами в діапазоні [-0.1, 0.1]
net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
net.init()

# Навчання мережі на вхідних та цільових даних
# epochs=500 - кількість епох
# show=100 - відображати помилку кожні 100 епох
# goal=0.01 - цільове значення помилки
error = net.train(input, target, epochs=500, show=100, goal=0.01)

# Прогнозування результатів на тих самих вхідних даних
output = net.sim(input)

# Візуалізація помилки навчання
pl.subplot(211)
pl.plot(error)
pl.xlabel('Номер епохи')
pl.ylabel('Помилка навчання (MSE за замовчуванням)')

# Візуалізація цільових значень та прогнозованих значень мережі
pl.subplot(212)
pl.plot(target.reshape(80))
pl.plot(output.reshape(80))
pl.legend(['Цільове значення навчання', 'Вихід мережі'])
pl.show()
