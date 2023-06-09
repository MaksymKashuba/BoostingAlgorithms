import matplotlib.pyplot as plt

algorithms = ['AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
accuracies = [0.8667280822969445, 0.8656533087670812, 0.8702594810379242, 0.8713342545677875, 0.8740979579302932]
durations = [7.12, 11.84, 5.35, 0.66, 3.73]

best_algorithm = algorithms[accuracies.index(max(accuracies))]

plt.figure(figsize=(10, 5))
colors = ['red' if alg != best_algorithm else 'green' for alg in algorithms]
plt.bar(algorithms, accuracies, color=colors)
plt.title('Точність алгоритмів бустінгу')
plt.xlabel('Алгоритм')
plt.ylabel('Точність')
plt.ylim(0, 1)
plt.show()


plt.figure(figsize=(10, 5))
best_algorithm = algorithms[durations.index(min(durations))]
colors = ['red' if alg != best_algorithm else 'green' for alg in algorithms]
plt.bar(algorithms, durations, color=colors)
plt.title('Час виконання алгоритмів бустінгу')
plt.xlabel('Алгоритм')
plt.ylabel('Час (секунди)')
plt.show()
