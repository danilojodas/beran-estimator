import numpy as np
import matplotlib.pyplot as plt
import os

sf_benk_A_Low = []
sf_benk_A_Medium = []
sf_benk_A_High = []
sf_benk_B_Low = []
sf_benk_B_Medium = []
sf_benk_B_High = []
sf_benk_C_Low = []
sf_benk_C_Medium = []
sf_benk_C_High = []

N_n = []
data = [(sf_benk_A_Low, 'A', 'Low'), (sf_benk_A_Medium, 'A', 'Medium'), (sf_benk_A_High, 'A', 'High'), (sf_benk_B_Low, 'B', 'Low'), 
        (sf_benk_B_Medium, 'B', 'Medium'), (sf_benk_B_High, 'B', 'High'), (sf_benk_C_Low, 'C', 'Low'), (sf_benk_C_Medium, 'C', 'Medium'), 
        (sf_benk_C_High, 'C', 'High')]

for i in range(1, 51):
    for j in range(1, 101):
        path = 'test/results/dhsv/benk_{}_{}_results'.format(i, j)
        if os.path.exists(path):
            sf_benk_A_Low.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_A_Low.csv'.format(i, j), delimiter=','))
            sf_benk_A_Medium.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_A_Medium.csv'.format(i, j), delimiter=','))
            sf_benk_A_High.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_A_High.csv'.format(i, j), delimiter=','))
            sf_benk_B_Low.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_B_Low.csv'.format(i, j), delimiter=','))
            sf_benk_B_Medium.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_B_Medium.csv'.format(i, j), delimiter=','))
            sf_benk_B_High.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_B_High.csv'.format(i, j), delimiter=','))
            sf_benk_C_Low.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_C_Low.csv'.format(i, j), delimiter=','))
            sf_benk_C_Medium.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_C_Medium.csv'.format(i, j), delimiter=','))
            sf_benk_C_High.append(np.loadtxt('test/results/dhsv/benk_{}_{}_results/sf_C_High.csv'.format(i, j), delimiter=','))
            N_n.append((i, j))

plots_folder = 'benk_plots/dhsv_plots'
if (not os.path.exists(plots_folder)):
    os.makedirs(plots_folder)

colors = ['lightcoral', 'lightgray', 'peachpuff', 'khaki', 'paleturquoise', 'lightgreen', 'lightpink', 'plum', 
          'lightskyblue', 'sandybrown']

for (combo, oem, lda) in data:
    i = 0
    plt.figure(figsize=(20, 8))
    for sample in combo:
        plt.plot(sample[0], sample[1], label=f'BENK - {N_n[i][0]}/{N_n[i][1]}', color=colors[i])
        i = i+1
    opf = np.loadtxt('test/results/dhsv/opf_results/sf_opf_{}_{}.csv'.format(oem, lda), delimiter=',')
    opf_arc = np.loadtxt('test/results/dhsv/opf_arc_results/sf_opf_arc_{}_{}.csv'.format(oem, lda), delimiter=',')
    beran_plknn = np.loadtxt('test/results/dhsv/pl_knn_results/sf_plknn_{}_{}.csv'.format(oem, lda), delimiter=',')

    plt.plot(opf[0], opf[1], linestyle='dashed', label='Beran OPF-kNN', color='blue')
    plt.plot(opf_arc[0], opf_arc[1], linestyle='dashed', label='Beran OPF-kNN (Arc weights)', color='green')
    plt.plot(beran_plknn[0], beran_plknn[1], linestyle='dashed', label='Beran PL-kNN', color='red')
    plt.title('OEM {} - WC {}'.format(oem, lda), fontsize=28)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Survival probability', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(ncol=2)
    plt.savefig('{}/plot_benk_{}_{}.png'.format(plots_folder, oem, lda), dpi=300, bbox_inches='tight')
    plt.cla()