# import sys
# import pandas as pd
# import numpy as np

# sys.path.append(".")

# from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw
# from sksurv.datasets.base import get_x_y

# folders = ['coxphm', 'beran_plknn', 'rsf']

# # Train and test data
# train = pd.read_csv("test/DHSV_LDA_OEM_train.csv")
# test = pd.read_csv("test/DHSV_LDA_OEM_test.csv")

# # Name of the delta and time columns as list
# cols = ['delta', 'time']

# # Get the events as boolean
# event = np.where(test['delta'] == 1, True, False)

# # Get the structured array
# x_train, y_train = get_x_y(train, attr_labels=cols, pos_label=1, survival=True)
# x_test, y_test = get_x_y(test, attr_labels=cols, pos_label=1, survival=True)

# for folder in folders:
#     # Find the file that contains test in the name

#     # Read estimations
#     data = pd.read_csv(f"test/results3/{folder}/sf_test.csv", header=None)

#     # Read times
#     times = data.iloc[0,:].values.astype(int)
#     print('Times: ', times)

#     # Read survival function
#     sf = data.iloc[1:,:].values    

#     # Compute risk scores
#     risk_scores = np.array([-np.log(p) for p in sf])

#     # Get the risks for the last time
#     risk_scores = risk_scores[:,100]

#     # Evaluation time
#     eval_time = np.linspace(test['time'].min(), test['time'].max() - 1e-10, test['time'].shape[0])
#     surv = np.array([np.interp(eval_time, times, sf[i,:]) for i in range(len(sf))])

#     print('eval_time: ', eval_time.astype(int))

#     score = integrated_brier_score(y_train, y_test, surv, eval_time)

#     c_index = concordance_index_censored(event, eval_time, risk_scores)
#     c_index_ipcw = concordance_index_ipcw(y_train, y_test, risk_scores, eval_time)

#     print('Score for ', folder, ': ', score)
#     print('C-index for ', folder, ': ', c_index)
#     print('C-index IPCW for ', folder, ': ', c_index_ipcw)

import sys
import pandas as pd
import numpy as np

sys.path.append(".")

from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw
from sksurv.datasets.base import get_x_y

folders = {'Cox PHM':'cox_results', 'Beran PL-kNN':'pl_knn_results', 'RSF':'rsf_results', 'Beran OPF-kNN':'opf_results', 'Beran OPF-kNN (Arc weights)': 'opf_arc_results',
           'Benk N=1/n=28':'benk_1_28', 'Benk N=9/n=69':'benk_9_69', 'Benk N=10/n=49':'benk_10_49',
           'Benk N=13/n=55':'benk_13_55', 'Benk N=14/n=85':'benk_14_85', 'Benk N=20/n=9':'benk_20_9',
           'Benk N=25/n=65': 'benk_25_65', 'Benk N=26/n=30':'benk_26_30', 'Benk N=35/n=54':'benk_35_54',
           'Benk N=39/n=74':'benk_39_74'}

# Train and test data
train = pd.read_csv("test/DHSV_LDA_OEM_train.csv")
test = pd.read_csv("test/DHSV_LDA_OEM_test.csv")

# Name of the delta and time columns as list
cols = ['delta', 'time']

# Get the events as boolean
event = test['delta'].values == 1  # Boolean array

# Get the structured array
x_train, y_train = get_x_y(train, attr_labels=cols, pos_label=1, survival=True)
x_test, y_test = get_x_y(test, attr_labels=cols, pos_label=1, survival=True)

# 
metrics = []

for estimator,folder in folders.items():
    # Read estimations
    data = pd.read_csv(f"test/results3/{folder}/sf_test.csv", header=None)

    # Read times
    times = data.iloc[0, :].values.astype(float)  # Ensure it's float

    # Read survival function
    sf = data.iloc[1:, :].values    

    # Choose a meaningful evaluation time (e.g., median event time)
    t_star = np.median(test['time'])

    # Compute risk scores using survival probability at t_star
    risk_scores = np.array([-np.log(np.interp(t_star, times, sf[i, :])) for i in range(len(sf))])

    # Use actual event times from test set
    eval_time = np.linspace(test['time'].min(), test['time'].max() - 1e-10, test['time'].shape[0])

    # Interpolate survival function at actual test times
    surv = np.array([np.interp(eval_time, times, sf[i, :]) for i in range(len(sf))])

    # Compute metrics
    score = integrated_brier_score(y_train, y_test, surv, eval_time)
    c_index = concordance_index_censored(event, eval_time, risk_scores)
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, risk_scores, t_star)

    metrics.append([estimator, score, c_index[0], c_index_ipcw[0]])

    print(f"Score for {folder}: {score}")
    print(f"C-index for {folder}: {c_index[0]}")  # Extracting the first value of the tuple
    print(f"C-index IPCW for {folder}: {c_index_ipcw[0]}")

# Save metrics to a CSV file
df = pd.DataFrame(metrics, columns=['Estimator', 'IBS', 'C-index', 'C-index IPCW'], index=None)
df.to_csv('test/results3/metrics_2.csv', index=False,float_format='%.3f')
