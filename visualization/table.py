import matplotlib.pyplot as plt

algorithms = ['AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
accuracies = [0.8667280822969445, 0.8656533087670812, 0.8702594810379242, 0.8713342545677875, 0.8740979579302932]
durations = [7.12, 11.84, 5.35, 0.66, 3.73]

data = {'Алгоритм': algorithms, 'Точність': accuracies, 'Час виконання (сек)': durations}

table_data = [list(data.keys())]
table_data += list(zip(*data.values()))

fig, ax = plt.subplots(figsize=(8, 4))

table = plt.table(cellText=table_data,
                  loc='center')

table.scale(1, 1.5)

ax.axis('off')

plt.title('Результати алгоритмів бустінгу')
plt.show()
