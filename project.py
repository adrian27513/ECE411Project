import json
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(9000)
np.seterr(divide='ignore', invalid='ignore')


# Data Cleaning and Variable Coding
def get_df(sheets):
    df = pd.read_excel('FREEVAL_by_type.xlsx', sheet_name=sheets[0])
    for sheet in sheets:
        if sheet != sheets[0]:
            df_temp = pd.read_excel('FREEVAL_by_type.xlsx', sheet_name=sheet)
            df = pd.concat([df, df_temp], ignore_index=True)

    sum = ((df['Safety'] < 50) & (df['Safety'] >= 49.9)).sum()
    print("Sum:", sum)
    print("Length:", len(df.index))
    print("Percentage:", sum / len(df.index) * 100)
    df = df.drop(df[(df['Safety'] < 50) & (df['Safety'] >= 49.9)].index)
    safety = df['Safety']
    df = df.drop(columns=['Id', 'Safety', 'AADT_Profile'])
    df = pd.get_dummies(df, drop_first=True, dummy_na=False)
    original_column = df.columns
    df.columns = ["v" + str(s) for s in range(len(df.columns))]
    df['Safety'] = safety
    column_dict = dict(zip(original_column, df.columns))
    opposite_column_dict = dict(zip(df.columns, original_column))
    return df, column_dict, opposite_column_dict


data_df, cn, opposite_cn = get_df(['type0', 'type1', 'type2', 'type3', 'type4'])

with open("column_names.json", "w") as outfile:
    json.dump(cn, outfile)

with open("column_names_opposite.json", "w") as outfile:
    json.dump(opposite_cn, outfile)

data_df = data_df.sample(frac=1).reset_index(drop=True).fillna(0).replace(np.nan, 0)

data_len = len(data_df.index)
train_df = data_df.iloc[:int(data_len * 0.9)]
test_df = data_df.iloc[int(data_len * 0.9):]

data_df.to_csv('all_data.csv', index=False)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

###############################################################################

# sns.histplot(data=data_df, x="Safety")
# plt.show()

# data_df['Safety'] = np.log(np.add(data_df['Safety'],1))
# sns.histplot(data=data_df, x="Safety")
# plt.show()
# numerical = data_df.select_dtypes(['number'])
# numerical = numerical.dropna(axis='columns', how='all')

# sns_plot = sns.pairplot(numerical)
# print(sns_plot)
# sns_plot.savefig("pairplot.png")

# # sns.displot(df['Safety'])
# # plt.show()
# # df.boxplot(column=['Safety'])
# # plt.show()
# # print("starting")
# # sns_plot = sns.pairplot(numerical)
# #
# # sns_plot.savefig("pairplot0.png")
# #
# # plt.clf() # Clean parirplot figure from sns
#
# # print("Placeholders:", original_length - placeholder_length)
# # print("Percentage:", (original_length - placeholder_length)/original_length)
# graph_df = pd.DataFrame()
# graph_df['AC'] = df['Congestion_Level']
# graph_df['ZC'] = df[df['Safety'] == 0]['Congestion_Level']
# # graph_df['AT'] = df['turbulance level']
# # graph_df['ZT'] = df[df['Safety'] == 0]['turbulance level']
# # graph_df.boxplot(column=['AC', 'ZC'])
# # plt.show()
# # graph_df.boxplot(column=['AT', 'ZT'])
# # plt.show()
#
# graph_df['6C'] = df[df['Safety'] == 0.6]['Congestion_Level']
# # graph_df['6T'] = df[df['Safety'] == 0.6]['turbulance level']
# # graph_df.boxplot(column=['AC', 'ZC', '6C'])
# # plt.show()
# # graph_df.boxplot(column=['AT', 'ZT', '6T'])
# # plt.show()
#
# graph_df['4C'] = df[df['Safety'] == 0.4]['Congestion_Level']
# # graph_df['4T'] = df[df['Safety'] == 0.4]['turbulance level']
# # graph_df.boxplot(column=['AC', 'ZC', '4C'])
# # plt.show()
# # graph_df.boxplot(column=['AT', 'ZT', '4T'])
# # plt.show()
#
# print(graph_df['AC'].describe())
# print(graph_df['ZC'].describe())
# print(graph_df['6C'].describe())
# print(graph_df['4C'].describe())

# Statistical Analysis
train_data = pd.read_csv("train.csv")
train_len = len(train_data.index)

subset1 = train_data.iloc[:int(train_len * 0.2)]
subset2 = train_data.iloc[int(train_len * 0.2):int(train_len * 0.4)]
subset3 = train_data.iloc[int(train_len * 0.4):int(train_len * 0.6)]
subset4 = train_data.iloc[int(train_len * 0.6):int(train_len * 0.8)]
subset5 = train_data.iloc[int(train_len * 0.8):]

data = [subset1, subset2, subset3, subset4, subset5]

final_mse = []
final_mse_exp = []
final_features = []

MSE_thresh_final = []
MSE_exp_thresh_final = []
MSE_list_final = []
MSE_exp_list_final = []

trials = 501
for id, subset in enumerate(data):
    print("##########################################################################################")
    print("Fold: " + str(id))
    # K-Fold
    train = pd.concat([s for s in data if not s.equals(subset)])
    test = subset.copy()

    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.dropna(inplace=True)

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.dropna(inplace=True)

    # Transformation
    train['Safety'] = np.log(np.add(train['Safety'], 1))
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    train.dropna(inplace=True)

    test['Safety'] = np.log(np.add(test['Safety'], 1))
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    test.dropna(inplace=True)

    f = open('column_names.json')
    cn = json.load(f)
    f.close()

    train_X = train.drop(columns='Safety')
    train_y = train['Safety']

    test_X = test.drop(columns='Safety')
    test_y = test['Safety']
    test_y_exp = np.subtract(np.exp(test_y), 1)

    # Feature Selection using Select K Best
    print("Selecting Features on K Best")
    MSE_list = []
    MSE_exp_list = []
    for number in range(1, trials):
        # print(number)
        features_model = SelectKBest(score_func=f_regression, k=number)
        features_model.fit(train_X, train_y)
        features = features_model.get_feature_names_out(list(train_X.columns))

        my_formula = "Safety~" + "+".join(features)
        lm_fit = smf.ols(formula=my_formula, data=train, missing='drop').fit()
        predictions = lm_fit.predict(test.drop(columns='Safety'))
        predictions_exp = np.exp(predictions)
        MSE = np.sum(np.square(np.subtract(predictions, test_y))) / test_y.size
        MSE_exp = np.sum(np.square(np.subtract(predictions_exp, np.add(test_y_exp, -1)))) / test_y.size
        MSE_list.append(MSE)
        MSE_exp_list.append(MSE_exp)

    plt.plot(range(1, trials), MSE_list)
    plt.show()
    plt.clf()
    plt.plot(range(1, trials), MSE_exp_list)
    plt.show()
    plt.clf()

    MSE_list_final.append(MSE_list)
    MSE_exp_list_final.append(MSE_exp_list)

    print("Features Min: " + str(np.argmin(MSE_list)))
    print("Exp Features Min: " + str(np.argmin(MSE_exp_list)))

    features_model = SelectKBest(score_func=f_regression, k=np.argmin(MSE_exp_list))
    features_model.fit(train_X, train_y)
    features = features_model.get_feature_names_out(list(train_X.columns))

    my_formula = "Safety~" + "+".join(features)
    lm_fit = smf.ols(formula=my_formula, data=train, missing='drop').fit()
    predictions = lm_fit.predict(test.drop(columns='Safety'))
    predictions_exp = np.exp(predictions)
    MSE = np.sum(np.square(predictions - test_y)) / test_y.size
    MSE_exp = np.sum(np.square(np.subtract(predictions_exp, np.add(test_y_exp, -1)))) / test_y.size
    print(MSE)
    print(MSE_exp)

    print(features)

    new_train_X = train_X[features]
    my_formula = "Safety~" + "+".join(new_train_X.columns)
    lm_fit = smf.ols(formula=my_formula, data=train, missing='drop').fit()
    predictions = lm_fit.predict(test.drop(columns='Safety'))
    predictions_exp = np.exp(predictions)
    MSE = np.sum(np.square(np.subtract(predictions, test_y))) / test_y.size
    MSE_exp = np.sum(np.square(np.subtract(predictions_exp, np.add(test_y_exp, -1)))) / test_y_exp.size
    print(MSE)
    print(MSE_exp)
    out_features = list(new_train_X.columns)
    print(final_features)
    print(lm_fit.summary())

    # Final Stats
    final_features.append(out_features)
    final_mse.append(MSE)
    final_mse_exp.append(MSE_exp)

    # Feature Names
    feature_out = {}

    f = open('column_names_opposite.json')
    cn_opposite = json.load(f)
    f.close()

    for name in out_features:
        feature_out[name] = cn_opposite[name]

    with open("feature_names" + str(id) + ".json", "w") as outfile:
        json.dump(feature_out, outfile)

print("Final MSE:")
print(final_mse)
print("Final MSE Exp:")
print(final_mse_exp)
print("Final Features: ")
print(final_features)

print("MSE K-Best")
plt.plot(range(1, trials), MSE_list_final[0], label="Fold 1")
plt.plot(range(1, trials), MSE_list_final[1], label="Fold 2")
plt.plot(range(1, trials), MSE_list_final[2], label="Fold 3")
plt.plot(range(1, trials), MSE_list_final[3], label="Fold 4")
plt.plot(range(1, trials), MSE_list_final[4], label="Fold 5")
plt.legend()
plt.show()
plt.clf()

print("MSE K-Best Exp")
plt.plot(range(1, trials), MSE_exp_list_final[0], label="Fold 1")
plt.plot(range(1, trials), MSE_exp_list_final[1], label="Fold 2")
plt.plot(range(1, trials), MSE_exp_list_final[2], label="Fold 3")
plt.plot(range(1, trials), MSE_exp_list_final[3], label="Fold 4")
plt.plot(range(1, trials), MSE_exp_list_final[4], label="Fold 5")
plt.legend()
plt.show()
plt.clf()


# Prediction System Design
smallest_mse = np.argmin(final_mse_exp)
final_features_list = final_features[smallest_mse]
print("Feature ", str(smallest_mse))
feature_len = len(final_features_list)

train_data = pd.read_csv("train.csv")
train_data = train_data.drop(train_data[(train_data['Safety'] == 0)].index)
train_len = len(train_data.index)
print(train_len)

subset1 = train_data.iloc[:int(train_len * 0.2)]
subset2 = train_data.iloc[int(train_len * 0.2):int(train_len * 0.4)]
subset3 = train_data.iloc[int(train_len * 0.4):int(train_len * 0.6)]
subset4 = train_data.iloc[int(train_len * 0.6):int(train_len * 0.8)]
subset5 = train_data.iloc[int(train_len * 0.8):]

data = [subset1, subset2, subset3, subset4, subset5]

poisson = []
ols = []
mlp_20_a = []

poisson_model = None
ols_model = None
nn_model = None
for id, subset in enumerate(data):
    print("##########################################################################################")
    print("Fold: " + str(id))
    ## K-Fold
    train = pd.concat([s for s in data if not s.equals(subset)])
    test = subset.copy()

    X = train.drop(columns="Safety")[final_features_list]
    y = train['Safety']
    test_X = test.drop(columns="Safety")[final_features_list]
    test_y = test['Safety']

    train_data_log = np.log(np.add(train, 1))
    train_data_log.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    train_data_log.dropna(inplace=True)
    train_len = len(train_data_log.index)
    train_log = train_data_log.copy().iloc[:int(train_len * 0.9)]
    test_log = train_data_log.copy().iloc[int(train_len * 0.9):]
    X_log = train_log.drop(columns="Safety")[final_features_list]
    y_log = train_log['Safety']
    test_X_log = test_log.drop(columns="Safety")[final_features_list]
    test_y_log = test_log['Safety']

    print("GLM Poisson")
    print("---")
    formula = "Safety~" + "+".join(final_features_list)
    mod1 = poisson_model = smf.glm(formula=formula, data=train, family=sm.families.Poisson()).fit()
    predictions_p = mod1.predict(test_X)
    MSE = np.sqrt(np.sum(np.square(np.subtract(predictions_p, test_y))) / test_y.size)
    MAE = abs(np.sum(np.subtract(predictions_p, test_y))) / test_y.size
    poisson.append((MSE, MAE))

    print("OLS")
    print("---")
    lm_fit = ols_model = smf.ols(formula=formula, data=train_log, missing='drop').fit()
    predictions_o = lm_fit.predict(test_log.drop(columns='Safety'))
    predictions_exp = np.exp(predictions_o)
    MSE_exp = np.sqrt(np.sum(np.square(np.subtract(predictions_exp, np.add(test_y_log, -1)))) / test_y_log.size)
    MAE_exp = abs(np.sum((np.subtract(predictions_exp, np.add(test_y_log, -1))))) / test_y_log.size
    ols.append((MSE_exp, MAE_exp))

    print("NN 22")
    print("---")
    clf = MLPRegressor(solver='adam', hidden_layer_sizes=24, max_iter=99999999999, n_iter_no_change=10000,
                       early_stopping=True, verbose=False)
    clf.fit(X.values, y.values)
    nn_model = clf
    predictions_a = clf.predict(test_X.values)
    MSE = np.sqrt(np.sum(np.square(np.subtract(predictions_a, test_y))) / test_y.size)
    MAE = abs(np.sum(np.subtract(predictions_a, test_y))) / test_y.size
    mlp_20_a.append((MSE, MAE))

print(poisson)
print(ols)
print(mlp_20_a)

output = [poisson, ols, mlp_20_a]
name = ['poisson', 'ols', 'mlp_20_a']

for out_lists, name in zip(output, name):
    sum_mse = 0
    sum_mae = 0
    for results in out_lists:
        sum_mse += results[0]
        sum_mae += results[1]
    print(name)
    print(sum_mse)
    print(sum_mae)

# Prediction System Test
train = train_data.drop(train_data[(train_data['Safety'] == 0)].index)
test_X = train[final_features_list]
test_y = train['Safety']

model_predictions = poisson_model.predict(test_X)

print(len(model_predictions))
print(len(test_y.index))
MSE = np.sqrt(np.sum(np.square(np.subtract(model_predictions, test_y))) / test_y.size)
MAE = abs(np.sum(np.subtract(model_predictions, test_y))) / test_y.size
print("RMSE:", MSE)
print("MAE:", MAE)
