import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_dataset_ = pd.read_feather('../input/amexfeather/train_data.ftr')
# keep the latest statement features for each customer
train_dataset = train_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()

train_dataset.shape

del train_dataset_
gc.collect()

"""## Feature Analysis"""

for col in train_dataset.columns:
    if train_dataset[col].dtype == "category":
        print(col, train_dataset[col].dtype)

categorical_cols = ['D_63', 'D_64', 'D_66', 'D_68', 'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

numerical_cols = [col for col in train_dataset.columns if col not in categorical_cols + ["target"]]

print(f'Number of features: {len(train_dataset.columns)}')
print(f'Number of categorical features: {len(categorical_cols)}')
print(f'Number of continuos features: {len(numerical_cols)}')

# figure out columns to drop by analyzing null value counts
for col in train_dataset.columns:
    if train_dataset[col].isna().sum() > train_dataset.shape[0]*0.75:
        print(col, train_dataset[col].isna().sum())

train_dataset = train_dataset.drop(['D_66','D_42','D_49','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142','S_2'], axis=1)

numerical_cols

# fill missing values of numerical features
null_numeric_cols = np.array(['P_2','S_3','B_2','D_41','D_43','B_3','D_44','D_45','D_46','D_48','D_50','D_53','S_7','D_56','S_9','B_6','B_8','D_52','P_3','D_54','D_55','B_13','D_59','D_61','B_15','D_62','B_16','B_17','D_77','B_19','B_20','D_69','B_22','D_70','D_72','D_74','R_7','B_25','B_26','D_78','D_79','D_80','B_27','D_81','R_12','D_82','D_105','S_27','D_83','R_14','D_84','D_86','R_20','B_33','D_89','D_91','S_22','S_23','S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_37','R_27','D_109','D_112','B_40','D_113','D_115','D_118','D_119','D_121','D_122','D_123','D_124','D_125','D_128','D_129','B_41','D_130','D_131','D_133','D_139','D_140','D_141','D_143','D_144','D_145'])

for col in null_numeric_cols:
    train_dataset[col] = train_dataset[col].fillna(train_dataset[col].median())

# fill missing values of categorical features
null_categorical_cols = np.array(['D_68','B_30','B_38','D_64','D_114','D_116','D_117','D_120','D_126'])

for col in null_categorical_cols:
    train_dataset[col] =  train_dataset[col].fillna(train_dataset[col].mode()[0])

"""### Apply the same operations to the test dataset"""

test_dataset_ = pd.read_feather('../input/amexfeather/test_data.ftr')
# Keep the latest statement features for each customer
test_dataset = test_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()

del test_dataset_
gc.collect()

test_dataset = test_dataset.drop(['S_2','D_42','D_49','D_66','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)

null_numeric_cols = np.array(['P_2','S_3','B_2','D_41','D_43','B_3','D_44','D_45','D_46','D_48','D_50','D_53','S_7','D_56','S_9','S_12','S_17','B_6','B_8','D_52','P_3','D_54','D_55','B_13','D_59','D_61','B_15','D_62','B_16','B_17','D_77','B_19','B_20','D_69','B_22','D_70','D_72','D_74','R_7','B_25','B_26','D_78','D_79','D_80','B_27','D_81','R_12','D_82','D_105','S_27','D_83','R_14','D_84','D_86','R_20','B_33','D_89','D_91','S_22','S_23','S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_37','R_27','D_109','D_112','B_40','D_113','D_115','D_118','D_119','D_121','D_122','D_123','D_124','D_125','D_128','D_129','B_41','D_130','D_131','D_133','D_139','D_140','D_141','D_143','D_144','D_145'])

for col in null_numeric_cols:
    test_dataset[col] = test_dataset[col].fillna(train_dataset[col].median())

null_categorical_cols = np.array(['D_68','B_30','B_38','D_114','D_116','D_117','D_120','D_126'])

for col in null_categorical_cols:
    test_dataset[col] =  test_dataset[col].fillna(train_dataset[col].mode()[0])

# D_66 was removed when counting null value percentage
categorical_cols.remove('D_66')

print(test_dataset.shape)
print(train_dataset.shape)

# reomve highly correlated features
train_dataset_features = train_dataset.drop(["target"],axis=1)

corr_matrix = train_dataset_features.corr()
correlated_cols = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if(corr_matrix.iloc[i, j] > 0.9):
            col_name = corr_matrix.columns[i]
            correlated_cols.add(col_name)
correlated_cols

train_dataset = train_dataset.drop(correlated_cols, axis=1)
test_dataset = test_dataset.drop(correlated_cols, axis=1)

print(test_dataset.shape)
print(train_dataset.shape)

"""## Downsample Dataset (FOR SVM)"""

no_default_customers = train_dataset[train_dataset["target"] == 0]
default_customers  = train_dataset[train_dataset["target"] == 1]
print(no_default_customers.shape)
print(default_customers.shape)

no_default_customers_down_sampled = no_default_customers.sample(n = 118828)

train_dataset = pd.concat([default_customers,no_default_customers_down_sampled], axis=0 )

train_dataset = train_dataset.sample(frac = 1)

"""## Extract Target Variable"""

train_Y = train_dataset['target']
train_dataset = train_dataset.drop(['target'],1)

print(train_dataset.shape)
print(test_dataset.shape)

for col in categorical_cols:
    train_dataset[col] = train_dataset[col].astype(str)
for col in categorical_cols:
    test_dataset[col] = test_dataset[col].astype(str)

train_X = pd.get_dummies(train_dataset, columns = categorical_cols)
test_X = pd.get_dummies(test_dataset, columns = categorical_cols)

test_X = test_X.reindex(columns = train_X.columns, fill_value=0)

print(train_X.shape)
print(test_X.shape)

"""## Splitting Dataset"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=6)

# user SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

"""## Model Building"""

# full SVM model
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_X)
train_X = scaling.transform(train_X)
test_X = scaling.transform(test_X)

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(train_X, train_Y.ravel())

predictions = clf._predict_proba_lr(test_X)

# full KNN model
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(train_X, y)

knn_pred = knn_model.predict_proba(test_X)
predictions = knn_pred[:,1]

# test KNN model
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
X_train, y_train = oversample.fit_resample(X_train, y_train)  ## upsample the dataset to balance the classes
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict_proba(X_test)
predictions = knn_pred[:,1]

# full CatBoost Classifier model
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import Pool, cv

train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features = categorical_cols)

final_model = CatBoostClassifier(verbose=1000,  cat_features=categorical_cols)

final_model.fit(train_dataset)

predictions = final_model.predict_proba(test_dataset)

# full XGBoostClassifier model
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators = 1000,
                      seed = 6,
                     eta = 0.05,
                     max_depth = 5,
                     colsample_bytree = 0.9,
                      subsample = 0.9,
                     objective = "reg:logistic",
                     )

# user SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

train_X, train_Y = oversample.fit_resample(train_X, train_Y)

model.fit(train_X, train_Y)

predictions = model.predict_proba(test_X)

# test XGBoostClassifier model
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators = 500,
                      seed = 6,
                     eta = 0.005,
                     max_depth = 5,
                     colsample_bytree = 0.8,
                      subsample = 0.8,
                     objective = "reg:logistic",
                     )

# user SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

X_train, y_train = oversample.fit_resample(X_train, y_train)

model.fit(X_train, y_train)

predictions = model.predict_proba(X_test)

"""## MOdel Evaluation"""

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

y_true = y_test.to_frame(name = 'target')
y_true = y_true.reset_index(drop=True)

y_pred = pd.DataFrame(predictions[:,1], columns = ['prediction'])

print(amex_metric(y_true, y_pred))

"""## Save Submission File"""

predictions = predictions[:,1]

sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions})
output.to_csv('submission.csv', index=False)