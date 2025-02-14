#Model using RandomForestRegressor, ADA, SVM (SVR), LGBM.
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

################################################################################
##########################FUNCTION DEFINITIONS##################################
def frequency_in_top_k(df, k):
    frequency_dict = {column: 0 for column in df.columns}

    for index, row in df.iterrows():
        #sort the row to get the top k model names
        top_k_models = row.sort_values(ascending=False).head(k).index

        #update the frequency count for each model in the top k
        for model in top_k_models:
            if model in frequency_dict:
                frequency_dict[model] +=1

    return frequency_dict

################################################################################
#############################DATA PREPROCESSING#################################

column_names = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
]


prob_ada_column_names = [
    'ADA-0', 'ADA-1', 'ADA-2', 'ADA-3', 'ADA-4', 'ADA-5', 'ADA-6', 'ADA-7', 'ADA-8'
]

prob_knn_column_names = [
    'KNN-0', 'KNN-1', 'KNN-2', 'KNN-3', 'KNN-4', 'KNN-5'
]

prob_lgbm_column_names = [
    'LGBM-0', 'LGBM-1', 'LGBM-2', 'LGBM-3'
]

prob_dnn_column_names = [
    'DNN-0', 'DNN-1', 'DNN-2', 'DNN-3', 'DNN-4', 'DNN-5', 'DNN-6', 'DNN-7', 'DNN-8'
]

prob_mlp_column_names = [
    'MLP-0', 'MLP-1', 'MLP-2', 'MLP-3', 'MLP-4', 'MLP-5', 'MLP-6', 'MLP-7', 'MLP-8'
]

prob_rf_column_names = [
    'RF-0', 'RF-1', 'RF-2', 'RF-3', 'RF-4', 'RF-5', 'RF-6', 'RF-7', 
    'RF-8', 'RF-9', 'RF-10', 'RF-11', 'RF-12', 'RF-13', 'RF-14', 'RF-15',
    'RF-16', 'RF-17'
]

prob_sgd_column_names = [
    'SGD-0', 'SGD-1', 'SGD-2', 'SGD-3', 'SGD-4', 'SGD-5', 'SGD-6', 'SGD-7', 'SGD-8'
]

prob_output_column_names = [
    'ADA-0', 'ADA-1', 'ADA-2', 'ADA-3', 'ADA-4', 'ADA-5', 'ADA-6', 'ADA-7', 'ADA-8',
    'KNN-0', 'KNN-1', 'KNN-2', 'KNN-3', 'KNN-4', 'KNN-5',
    'LGBM-0', 'LGBM-1', 'LGBM-2', 'LGBM-3',
    'DNN-0', 'DNN-1', 'DNN-2', 'DNN-3', 'DNN-4', 'DNN-5', 'DNN-6', 'DNN-7', 'DNN-8',
    'MLP-0', 'MLP-1', 'MLP-2', 'MLP-3', 'MLP-4', 'MLP-5', 'MLP-6', 'MLP-7', 'MLP-8',
    'RF-0', 'RF-1', 'RF-2', 'RF-3', 'RF-4', 'RF-5', 'RF-6', 'RF-7', 
    'RF-8', 'RF-9', 'RF-10', 'RF-11', 'RF-12', 'RF-13', 'RF-14', 'RF-15',
    'RF-16', 'RF-17',
    'SGD-0', 'SGD-1', 'SGD-2', 'SGD-3', 'SGD-4', 'SGD-5', 'SGD-6', 'SGD-7', 'SGD-8'
]

#Beginning of Test Data Setup
test_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\cicids_test.csv"
test_dataset = pd.read_csv(test_path)
label_and_istrain = test_dataset.columns[:-2]
test_dataset = pd.read_csv(test_path, usecols=label_and_istrain)
print("Shape of test_dataset: ", test_dataset.shape)
print(test_dataset)


#Beginning of probability usage

knn_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\knn_selected_probabilities_new.csv"
knn_probabilities = pd.read_csv(knn_prob_path, header=None, names=prob_knn_column_names, dtype='object')
knn_probabilities = knn_probabilities.iloc[1:]#removing label
knn_probabilities = knn_probabilities.reset_index(drop=True)

prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\ada_cicids_probabilities.csv"
ada_probabilities = pd.read_csv(prob_path, header=None, names=prob_ada_column_names, dtype='object')
ada_probabilities = ada_probabilities.iloc[1:] #removing label
ada_probabilities = ada_probabilities.head(len(knn_probabilities))
ada_probabilities = ada_probabilities.reset_index(drop=True)


lgbm_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\lgbm_hyperparameter_probabilities.csv"
lgbm_probabilities = pd.read_csv(lgbm_prob_path, header=None, names=prob_lgbm_column_names, dtype='object')
lgbm_probabilities = lgbm_probabilities.iloc[1:]#removing label
lgbm_probabilities = lgbm_probabilities.reset_index(drop=True)


dnn_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\dnn_selected_probabilities_cicidsv1.csv"
dnn_probabilities = pd.read_csv(dnn_prob_path, header=None, names=prob_dnn_column_names, dtype='object')
dnn_probabilities = dnn_probabilities.iloc[1:]#removing label
dnn_probabilities = dnn_probabilities.reset_index(drop=True)

mlp_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\mlp_selected_probabilities.csv"
mlp_probabilities = pd.read_csv(mlp_prob_path, header=None, names=prob_mlp_column_names, dtype='object')
mlp_probabilities = mlp_probabilities.iloc[1:]#removing label
mlp_probabilities = mlp_probabilities.reset_index(drop=True)

rf_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\rf_selected_probabilities.csv"
rf_probabilities = pd.read_csv(rf_prob_path, header=None, names=prob_rf_column_names, dtype='object')
rf_probabilities = rf_probabilities.iloc[1:]#removing label
rf_probabilities = rf_probabilities.reset_index(drop=True)

sgd_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\sgd_selected_probabilities.csv"
sgd_probabilities = pd.read_csv(sgd_prob_path, header=None, names=prob_sgd_column_names, dtype='object')
sgd_probabilities = sgd_probabilities.iloc[1:]#removing label
sgd_probabilities = sgd_probabilities.reset_index(drop=True)

combined_probabilities = pd.concat([ada_probabilities, knn_probabilities, lgbm_probabilities, dnn_probabilities, mlp_probabilities,
                                    rf_probabilities, sgd_probabilities],axis = 1) 
print("Shape of combined probabilities before na imputation drop: ", combined_probabilities.shape)
#Imputation necessary to take care of large amount of NaN values within dataset
mean_imputer = SimpleImputer(strategy='mean')
mean_imputed_probabilities = pd.DataFrame(mean_imputer.fit_transform(combined_probabilities), columns=prob_output_column_names)
print(mean_imputed_probabilities)
print("shape of Combined probabilites after imputation: ", mean_imputed_probabilities.shape)
#removing infinity values
mean_imputed_probabilities[np.isinf(mean_imputed_probabilities)] = np.finfo(np.float32).max

scaler = QuantileTransformer(output_distribution='uniform')
#combined_probabilities = scaler.fit_transform(mean_imputed_probabilities)
combined_probabilities = scaler.fit_transform(mean_imputed_probabilities)
# Save the combined data to a new CSV file
combined_probabilities = pd.DataFrame(combined_probabilities, columns = prob_output_column_names)
print(combined_probabilities)
print("shape of Combined probabilites after imputation and scaling: ", combined_probabilities.shape)
combined_probabilities.to_csv('combined_probabilities.csv', index=False)
print(combined_probabilities)
print("Shape of combined probabilities after imputation, scaling, and matching test: ", combined_probabilities.shape)
################################################################################
#############################TRAINING###########################################
X_train, X_test, Y_train, Y_test = train_test_split(test_dataset, combined_probabilities, test_size=.2, random_state=42)
Y_train_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Y_train_CICIDS.csv"
Y_train = pd.DataFrame(Y_train)
Y_train.to_csv(Y_train_prob_path, index=False)
Y_test_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Y_test_CICIDS.csv"
Y_test = pd.DataFrame(Y_test)
Y_test.to_csv(Y_test_prob_path, index=False)

#RFTRAIN
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale= scaler.fit_transform(X_test)
standard_scaler = StandardScaler()
X_train_scale_standard = standard_scaler.fit_transform(X_train_scale)
X_test_scale_standard = standard_scaler.fit_transform(X_test_scale)
regressor = RandomForestRegressor(random_state=42)
multioutput_regressor_RF = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_RF.fit(X_train_scale_standard, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("RF Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#ADATRAIN
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale= scaler.fit_transform(X_test)
standard_scaler = StandardScaler()
X_train_scale_standard = standard_scaler.fit_transform(X_train_scale)
X_test_scale_standard = standard_scaler.fit_transform(X_test_scale)
regressor = AdaBoostRegressor(random_state=42)
multioutput_regressor_ADA = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_ADA.fit(X_train_scale_standard, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("AdaBoostRegressor Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#SVMTRAIN
scaler = MinMaxScaler()
X_train_SVM = X_train[:5000]
X_test_SVM = X_train[:5000]
X_train_scale_SVM = scaler.fit_transform(X_train_SVM)
X_test_scale_SVM= scaler.fit_transform(X_test_SVM)
standard_scaler = StandardScaler()
X_train_scale_standard = standard_scaler.fit_transform(X_train_scale_SVM)
X_test_scale_standard = standard_scaler.fit_transform(X_test_scale_SVM)
regressor = SVR(kernel='rbf')
multioutput_regressor_SVM = MultiOutputRegressor(regressor)
start_train_time = time.time()
Y_train_SVM = Y_train[:5000]
multioutput_regressor_SVM.fit(X_train_scale_standard, Y_train_SVM)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("SVR Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#LGBMTRAIN
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale= scaler.fit_transform(X_test)
standard_scaler = StandardScaler()
X_train_scale_standard = standard_scaler.fit_transform(X_train_scale)
X_test_scale_standard = standard_scaler.fit_transform(X_test_scale)
regressor = LGBMRegressor(random_state=42)
multioutput_regressor_LGBM = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_LGBM.fit(X_train_scale_standard, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("LGBMRegressor Model Trained: \n")
print("\nTime it took to train model: ", training_time)

###################################################################################
#############################RF TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_RF.predict(X_test_scale_standard)
end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("RF Model Tested: \n")
print("\nTime it took to test model: ", testing_time)

Total_time = (training_time+testing_time)
print("\nTotal time taken for model training and testing: ", Total_time)

Calculation_time = Total_time/(22544)
print("\nCalculation time for one sample: ", Calculation_time)

mse = mean_squared_error(Y_test, y_pred)
print(f'\nMean Squared Error: {mse}\n\n\n')

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_RF.csv"
df_y_pred = pd.DataFrame(y_pred, columns = prob_output_column_names)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
df_y_pred = pd.DataFrame(df_y_pred, columns = prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_RF.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))
##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_1_RF.csv"
topk_1_RF = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_RF])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_5_RF.csv"
topk_5_RF = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_RF])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_10_RF.csv"
topk_10_RF = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_RF])
df_topk_10.to_csv(topk_10, index=False)
####################################################################################
#############################ADA TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_ADA.predict(X_test_scale_standard)
end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("ADABoostRegressor Model Tested: \n")
print("\nTime it took to test model: ", testing_time)

Total_time = (training_time+testing_time)
print("\nTotal time taken for model training and testing: ", Total_time)

Calculation_time = Total_time/(22544)
print("\nCalculation time for one sample: ", Calculation_time)

mse = mean_squared_error(Y_test, y_pred)
print(f'\nMean Squared Error: {mse}\n\n\n')

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_ADA.csv"
df_y_pred = pd.DataFrame(y_pred, columns = prob_output_column_names)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
df_y_pred = pd.DataFrame(df_y_pred, columns = prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_ADA.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))
##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_1_ADA.csv"
topk_1_RF = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_RF])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_5_ADA.csv"
topk_5_RF = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_RF])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_10_ADA.csv"
topk_10_RF = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_RF])
df_topk_10.to_csv(topk_10, index=False)
####################################################################################
#############################SVM TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_SVM.predict(X_test_scale_standard)
end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("SVM Model Tested: \n")
print("\nTime it took to test model: ", testing_time)

Total_time = (training_time+testing_time)
print("\nTotal time taken for model training and testing: ", Total_time)

Calculation_time = Total_time/(22544)
print("\nCalculation time for one sample: ", Calculation_time)

mse = mean_squared_error(Y_test, y_pred)
print(f'\nMean Squared Error: {mse}\n\n\n')

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_SVM.csv"
df_y_pred = pd.DataFrame(y_pred, columns = prob_output_column_names)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
df_y_pred = pd.DataFrame(df_y_pred, columns = prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_SVM.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))
##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_1_SVM.csv"
topk_1_RF = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_RF])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_5_SVM.csv"
topk_5_RF = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_RF])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_10_SVM.csv"
topk_10_RF = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_RF])
df_topk_10.to_csv(topk_10, index=False)
####################################################################################
#############################LGBM TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_LGBM.predict(X_test_scale_standard)
end_test_time = time.time()
testing_time = end_test_time - start_test_time
print("LGBM Model Tested: \n")
print("\nTime it took to test model: ", testing_time)

Total_time = (training_time+testing_time)
print("\nTotal time taken for model training and testing: ", Total_time)

Calculation_time = Total_time/(22544)
print("\nCalculation time for one sample: ", Calculation_time)

mse = mean_squared_error(Y_test, y_pred)
print(f'\nMean Squared Error: {mse}\n\n\n')

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_LGBM.csv"
df_y_pred = pd.DataFrame(y_pred, columns=prob_output_column_names)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
df_y_pred = pd.DataFrame(df_y_pred, columns=prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_probability_output_LGBM.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))
##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_1_LGBM.csv"
topk_1_RF = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_RF])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_5_LGBM.csv"
topk_5_RF = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_RF])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH CICIDS\Results\CICIDS_topk_10_LGBM.csv"
topk_10_RF = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_RF])
df_topk_10.to_csv(topk_10, index=False)