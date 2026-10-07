import numpy as np
import matplotlib.pyplot as plt
import os

one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []

N_n = []
data = [one, two, three, four, five, six, seven, eight, nine]

for i in range(1, 51):
    for j in range(1, 101):
        path = 'test/results/prostate_cancer/benk_{}_{}_results'.format(i, j)
        if os.path.exists(path):
            one.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_1.csv'.format(i, j), delimiter=','))
            two.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_2.csv'.format(i, j), delimiter=','))
            three.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_3.csv'.format(i, j), delimiter=','))
            four.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_4.csv'.format(i, j), delimiter=','))
            five.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_5.csv'.format(i, j), delimiter=','))
            six.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_6.csv'.format(i, j), delimiter=','))
            seven.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_7.csv'.format(i, j), delimiter=','))
            eight.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_8.csv'.format(i, j), delimiter=','))
            nine.append(np.loadtxt('test/results/prostate_cancer/benk_{}_{}_results/sf_9.csv'.format(i, j), delimiter=','))
            N_n.append((i, j))

plots_folder = 'benk_plots/prostate_cancer_plots'
if (not os.path.exists(plots_folder)):
    os.makedirs(plots_folder)

colors = ['lightcoral', 'lightgray', 'peachpuff', 'khaki', 'paleturquoise', 'lightgreen', 'lightpink', 'plum', 
          'lightskyblue', 'sandybrown']

for j, sample in enumerate(data):
    i = 0
    plt.figure(figsize=(20, 8))
    for combo in sample:
        plt.plot(combo[0], combo[1], label=f'BENK - {N_n[i][0]}/{N_n[i][1]}', color=colors[i])
        i = i+1
    opf = np.loadtxt('test/results/prostate_cancer/opf_results/sf_{}.csv'.format(j+1), delimiter=',')
    opf_arc = np.loadtxt('test/results/prostate_cancer/opf_arc_results/sf_{}.csv'.format(j+1), delimiter=',')
    beran_plknn = np.loadtxt('test/results/prostate_cancer/pl_knn_results/sf_{}.csv'.format(j+1), delimiter=',')

    plt.plot(opf[0], opf[1], linestyle='dashed', label='Beran OPF-kNN', color='blue')
    plt.plot(opf_arc[0], opf_arc[1], linestyle='dashed', label='Beran OPF-kNN (Arc weights)', color='green')
    plt.plot(beran_plknn[0], beran_plknn[1], linestyle='dashed', label='Beran PL-kNN', color='red')
    plt.title('Plot {}'.format(j), fontsize=28)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Survival probability', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend()
    plt.savefig('{}/{}.png'.format(plots_folder, j), dpi=300)
    plt.cla()