import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import datasets


data = pd.read_csv('FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binaryrom string "Yes"/"No" to binary
# copy all int64 data type to feature_names
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



# We will look at SHAP values for a single row of the dataset
# (we arbitrarily chose row 5). 
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
# invert rows, columns to columns, rows
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
#For context, we'll look at the raw predictions before looking at the SHAP values
my_model.predict_proba(data_for_prediction_array)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
### Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# It's cumbersome to review raw arrays, but the shap package has a nice way to visualize the results.
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

#Here is an example using KernelExplainer to get similar results.
#The results aren't identical because kernelExplainer gives an approximate result.
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
### Calculate Shap values
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)




# or

# **Calculate and show Shap Values for One Prediction:**
# ```
# import shap  # package used to calculate Shap values

# data_for_prediction = val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired

# # Create object that can calculate shap values
# explainer = shap.TreeExplainer(my_model)
# shap_values = explainer.shap_values(data_for_prediction)
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)