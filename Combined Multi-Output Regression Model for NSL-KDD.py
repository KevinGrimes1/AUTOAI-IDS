#Model using RandomForestRegressor, ADA, SVM (SVR), LGBM.
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer
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
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_error_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'severity_score'
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

categorical_columns = ["protocol_type", "service", "flag"]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

#Beginning of Test Data Setup
test_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\KDDTest+.txt"
test_dataset = pd.read_csv(test_path, header=None, names=column_names)
samples_test = test_dataset.drop('label', axis=1)

print("samples before encoding", samples_test)

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# Fit and transform the categorical columns
encoded_columns = pd.DataFrame(encoder.fit_transform(samples_test[categorical_columns]))

# Get the names of the encoded columns
encoded_columns.columns = encoder.get_feature_names_out(categorical_columns)

# Concatenate the original DataFrame with the encoded columns
data_encoded = pd.concat([samples_test.drop(categorical_columns, axis=1), encoded_columns], axis=1)

print("samples after encoding", data_encoded)

#Beginning of probability usage
prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\selected_probabilities_ada.csv"
ada_probabilities = pd.read_csv(prob_path, header=None, names=prob_ada_column_names)
ada_probabilities = ada_probabilities.loc[1:] #removing label

knn_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\knn_selected_probabilities.csv"
knn_probabilities = pd.read_csv(knn_prob_path, header=None, names=prob_knn_column_names)
knn_probabilities = knn_probabilities.loc[1:]#removing label

lgbm_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-kDD\lgbm_hyperparameter_probabilities.csv"
lgbm_probabilities = pd.read_csv(lgbm_prob_path, header=None, names=prob_lgbm_column_names)
lgbm_probabilities = lgbm_probabilities.loc[1:]#removing label
lgbm_probabilities = lgbm_probabilities.head(len(knn_probabilities))

dnn_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\dnn_selected_probabilities.csv"
dnn_probabilities = pd.read_csv(dnn_prob_path, header=None, names=prob_dnn_column_names)
dnn_probabilities = dnn_probabilities.loc[1:]#removing label

mlp_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\\ECE RESEARCH NSL-KDD\mlp_selected_probabilities.csv"
mlp_probabilities = pd.read_csv(mlp_prob_path, header=None, names=prob_mlp_column_names)
mlp_probabilities = mlp_probabilities.loc[1:]#removing label

rf_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\rf_selected_probabilities.csv"
rf_probabilities = pd.read_csv(rf_prob_path, header=None, names=prob_rf_column_names)
rf_probabilities = rf_probabilities.loc[1:]#removing label

sgd_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\sgd_selected_probabilities.csv"
sgd_probabilities = pd.read_csv(sgd_prob_path, header=None, names=prob_sgd_column_names)
sgd_probabilities = sgd_probabilities.loc[1:]#removing label

combined_probabilities = pd.concat([ada_probabilities, knn_probabilities, lgbm_probabilities, dnn_probabilities,
                                     mlp_probabilities, rf_probabilities, sgd_probabilities],axis = 1) 
# Save the combined data to a new CSV file
combined_probabilities.to_csv('combined_probabilities.csv', index=False)
print("Combined probabilites: ", combined_probabilities.shape)

################################################################################
#############################TRAINING###########################################
X_train, X_test, Y_train, Y_test = train_test_split(data_encoded, combined_probabilities, test_size=.2, random_state=42)
#RFTRAIN
Y_train_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\Y_train_NSL-KDD.csv"
Y_train = pd.DataFrame(Y_train)
Y_train.to_csv(Y_train_prob_path, index=False)
Y_test_prob_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\Y_test_NSL-KDD.csv"
Y_test = pd.DataFrame(Y_test)
Y_test.to_csv(Y_test_prob_path, index=False)

regressor = RandomForestRegressor(random_state=42)
multioutput_regressor_RF = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_RF.fit(X_train, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("RF Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#ADATRAIN
regressor = AdaBoostRegressor(random_state=42)
multioutput_regressor_ADA = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_ADA.fit(X_train, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("AdaBoostRegressor Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#SVMTRAIN
X_train_scale =preprocessing.scale(X_train)
X_test_scale=preprocessing.scale(X_test)
regressor = SVR(kernel='rbf')
multioutput_regressor_SVM = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_SVM.fit(X_train_scale, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("SVR Model Trained: \n")
print("\nTime it took to train model: ", training_time)

#LGBMTRAIN
regressor = LGBMRegressor(random_state=42)
multioutput_regressor_LGBM = MultiOutputRegressor(regressor)
start_train_time = time.time()
multioutput_regressor_LGBM.fit(X_train, Y_train)
end_train_time = time.time()
training_time = end_train_time - start_train_time
print("LGBMRegressor Model Trained: \n")
print("\nTime it took to train model: ", training_time)

###################################################################################
#############################RF TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_RF.predict(X_test)
#TODO:// normalization of y_pred values from 0-1.  
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

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_RF.csv"
pred_scaled_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\scaled_probability_output_RF.csv"
df_y_pred = pd.DataFrame(y_pred, columns = prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
#Convert to dataframe
df_y_pred = pd.DataFrame(df_y_pred, columns=prob_output_column_names)

df_y_pred.to_csv(pred_scaled_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_RF.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))

ones_count_dict = {}
# Iterate over each column in the DataFrame
for column in encoded_output.columns:
    # Count the occurrences of 1 in the column
    ones_count = (encoded_output[column] == 1).sum()
    # Store the count of 1's in the dictionary
    ones_count_dict[column] = ones_count

# Convert the dictionary to a DataFrame
ones_count_df = pd.DataFrame.from_dict(ones_count_dict, orient='index',columns=['Frequency_count'])
# Sort the DataFrame by the 'ones_count' column in descending order
sorted_df = ones_count_df.sort_values(by='Frequency_count', ascending=False)

# Select the top 5 rows
top_5 = sorted_df.head(10)

# Save the top 5 rows to a new CSV file
top_5.to_csv('APS_top_5_frequency_count_RF.csv')

# Divide each value in the 'Frequency count' column by 4510
sorted_df['Accuracy_per_sample'] = sorted_df['Frequency_count'] / 4510
# Save the modified DataFrame to a new CSV file
sorted_df.to_csv('Accuracy_per_sample_RF.csv', index=False)

##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_1_RF.csv"
topk_1_RF = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_RF])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_5_RF.csv"
topk_5_RF = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_RF])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_10_RF.csv"
topk_10_RF = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_RF])
df_topk_10.to_csv(topk_10, index=False)
####################################################################################
#############################ADA TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_ADA.predict(X_test)
#TODO:// normalization of y_pred values from 0-1. 
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

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_ADA.csv"
pred_scaled_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\scaled_probability_output_ADA.csv"
df_y_pred = pd.DataFrame(y_pred, columns=prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
#Convert to dataframe
df_y_pred = pd.DataFrame(df_y_pred, columns=prob_output_column_names)

df_y_pred.to_csv(pred_scaled_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_ADA.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))

ones_count_dict = {}
# Iterate over each column in the DataFrame
for column in encoded_output.columns:
    # Count the occurrences of 1 in the column
    ones_count = (encoded_output[column] == 1).sum()
    # Store the count of 1's in the dictionary
    ones_count_dict[column] = ones_count

# Convert the dictionary to a DataFrame
ones_count_df = pd.DataFrame.from_dict(ones_count_dict, orient='index',columns=['Frequency_count'])
# Sort the DataFrame by the 'ones_count' column in descending order
sorted_df = ones_count_df.sort_values(by='Frequency_count', ascending=False)

# Select the top 5 rows
top_5 = sorted_df.head(10)

# Save the top 5 rows to a new CSV file
top_5.to_csv('APS_top_5_frequency_count_ADA.csv')

# Divide each value in the 'Frequency count' column by 4510
sorted_df['Accuracy_per_sample'] = sorted_df['Frequency_count'] / 4510
# Save the modified DataFrame to a new CSV file
sorted_df.to_csv('Accuracy_per_sample_ADA.csv', index=False)

##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_1_ADA.csv"
topk_1_ADA = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_ADA])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_5_ADA.csv"
topk_5_ADA = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_ADA])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_10_ADA.csv"
topk_10_ADA = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_ADA])
df_topk_10.to_csv(topk_10, index=False)
####################################################################################
#############################SVM TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_SVM.predict(X_test_scale)
#TODO:// normalization of y_pred values from 0-1. 
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

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_SVM.csv"
pred_scaled_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\scaled_probability_output_SVM.csv"
df_y_pred = pd.DataFrame(y_pred, columns=prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
#Convert to dataframe
df_y_pred = pd.DataFrame(df_y_pred, columns=prob_output_column_names)


df_y_pred.to_csv(pred_scaled_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_SVM.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))

ones_count_dict = {}
# Iterate over each column in the DataFrame
for column in encoded_output.columns:
    # Count the occurrences of 1 in the column
    ones_count = (encoded_output[column] == 1).sum()
    # Store the count of 1's in the dictionary
    ones_count_dict[column] = ones_count

# Convert the dictionary to a DataFrame
ones_count_df = pd.DataFrame.from_dict(ones_count_dict, orient='index',columns=['Frequency_count'])
# Sort the DataFrame by the 'ones_count' column in descending order
sorted_df = ones_count_df.sort_values(by='Frequency_count', ascending=False)

# Select the top 5 rows
top_5 = sorted_df.head(10)

# Save the top 5 rows to a new CSV file
top_5.to_csv('APS_top_5_frequency_count_SVM.csv')

# Divide each value in the 'Frequency count' column by 4510
sorted_df['Accuracy_per_sample'] = sorted_df['Frequency_count'] / 4510
# Save the modified DataFrame to a new CSV file
sorted_df.to_csv('Accuracy_per_sample_SVM.csv', index=False)

##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_1_SVM.csv"
topk_1_SVM = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_SVM])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_5_SVM.csv"
topk_5_SVM = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_SVM])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_10_SVM.csv"
topk_10_SVM = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_SVM])
df_topk_10.to_csv(topk_10, index=False)

####################################################################################
#############################LGBM TESTING############################################
start_test_time = time.time()
y_pred = multioutput_regressor_LGBM.predict(X_test)
#TODO:// normalization of y_pred values from 0-1. 
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

pred_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_LGBM.csv"
pred_scaled_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\scaled_probability_output_LGBM.csv"
df_y_pred = pd.DataFrame(y_pred, columns=prob_output_column_names)
df_y_pred.to_csv(pred_path, index=False)

scaler = QuantileTransformer(output_distribution='uniform')
df_y_pred = scaler.fit_transform(df_y_pred)
#Convert to dataframe
df_y_pred = pd.DataFrame(df_y_pred, columns=prob_output_column_names)

df_y_pred.to_csv(pred_scaled_path, index=False)

####################################Accuracy per Sample##############################
output_path = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\probability_output_LGBM.csv"
output_dataset = pd.read_csv(output_path, header=0)
threshold =.5
encoded_output = output_dataset.apply(lambda x: (x > threshold).astype(int))

ones_count_dict = {}
# Iterate over each column in the DataFrame
for column in encoded_output.columns:
    # Count the occurrences of 1 in the column
    ones_count = (encoded_output[column] == 1).sum()
    # Store the count of 1's in the dictionary
    ones_count_dict[column] = ones_count

# Convert the dictionary to a DataFrame
ones_count_df = pd.DataFrame.from_dict(ones_count_dict, orient='index',columns=['Frequency_count'])
# Sort the DataFrame by the 'ones_count' column in descending order
sorted_df = ones_count_df.sort_values(by='Frequency_count', ascending=False)

# Select the top 5 rows
top_5 = sorted_df.head(10)

# Save the top 5 rows to a new CSV file
top_5.to_csv('APS_top_5_frequency_count_LGBM.csv')

# Divide each value in the 'Frequency count' column by 4510
sorted_df['Accuracy_per_sample'] = sorted_df['Frequency_count'] / 4510
# Save the modified DataFrame to a new CSV file
sorted_df.to_csv('Accuracy_per_sample_LGBM.csv', index=False)

##################################TOP MODEL PER SAMPLE##################################
topk_1 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_1_LGBM.csv"
topk_1_LGBM = frequency_in_top_k(output_dataset, 1)
df_topk_1 = pd.DataFrame([topk_1_LGBM])
df_topk_1.to_csv(topk_1, index=False)

topk_5 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_5_LGBM.csv"
topk_5_LGBM = frequency_in_top_k(output_dataset, 5)
df_topk_5 = pd.DataFrame([topk_5_LGBM])
df_topk_5.to_csv(topk_5, index=False)

topk_10 = r"C:\Users\kevin\Desktop\ECE RESEARCH\ECE RESEARCH NSL-KDD\topk_10_LGBM.csv"
topk_10_LGBM = frequency_in_top_k(output_dataset, 10)
df_topk_10 = pd.DataFrame([topk_10_LGBM])
df_topk_10.to_csv(topk_10, index=False)