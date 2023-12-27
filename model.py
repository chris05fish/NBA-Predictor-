import pandas as pd 
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("nba_games.csv", index_col=0)
#sort columns by date
df = df.sort_values("date")
#reset index to make 0,1,2 at top 
df = df.reset_index(drop=True)
#delete unnessesary/identicle rows
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]
#create a target column that is whether the team won or lost the next game they play
#function that shifts the won/lost column and pull it back one row
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)
#make null values in target column == 2
df["target"][pd.isnull(df["target"])] = 2
#convert win/loss from true/false to 1/0
df["target"] = df["target"].astype(int, errors="ignore")
#remove any columns with null values from dataset
nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
#create new dataframe with copy
df = df[valid_columns].copy()

#ridge regression for classification 
rr = RidgeClassifier(alpha=1)
#split data
split = TimeSeriesSplit(n_splits=3)
#feature selector to fit the model and prevent overfitting by making smaller data
sfs = SequentialFeatureSelector(rr, 
                                n_features_to_select=30, 
                                direction="forward",
                                cv=split,
                                n_jobs=1
                               )
#feature selection = train model to pick best features as it learns
#dont scale the removed columns because not numbers
#scale columns between 0 and 1 for ridge regression
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])
#fit data 
sfs.fit(df[selected_columns], df["target"])
#get list of columns from feature selector 
predictors = list(selected_columns[sfs.get_support()])

#split data up by seasons, start = 2 means needs at least two seasons to predict 
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        #all data before test season
        train = data[data["season"] < season]
        #current season to predict (ex: test set = 2018, train set = 2016, 2017)
        test = data[data["season"] == season]
        #train model
        model.fit(train[predictors], train["target"])
        #predict with model
        preds = model.predict(test[predictors])
        #convert to pandas to make easier to work with
        preds = pd.Series(preds, index=test.index)
        #put together values 
        combined = pd.concat([test["target"], preds], axis=1)
        #rename column 
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

#run model
predictions = backtest(df, rr, predictors)

#compare the two and find out percentage right
print(accuracy_score(predictions["actual"], predictions["prediction"]))
#55%

#make model better and more accurate, use baseline accuracy (ex. home team wins 60% of the time )
#shows percentage of wins by home team ~57%
df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

#compute rolling averages 
df_rolling = df[list(selected_columns) + ["won", "team", "season"]]
#for every row in df, find previous 10 games and calc mean for preformance 
def find_team_averages(team):
    # Select only numeric columns for rolling mean calculation
    numeric_cols = team.select_dtypes(include="number").columns
    rolling = team[numeric_cols].rolling(10).mean()
    return rolling
#avg preformance is specific to each team, only use rolling avg from that season 
df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

#combine rolling columns with regular
#rename _10 to end of column name for rolling avg
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
df = pd.concat([df, df_rolling], axis=1)
#drop rows with missing values
df = df.dropna()

#give info about next game, (home/away and opponent)
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

#indicate if team will be home or away next game
df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

#pull in stats about the opponent
#dataframe merge command (like inner sql join)
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])
#merge opponents info with our team info

#set of columns we dont want to pass into our model (all char or dates/not numbers)
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
#run model on new dataframe with new information on next games 
sfs.fit(full[selected_columns], full["target"])
#feature selector gets best features to fit the model

#list of predictors 
predictors = list(selected_columns[sfs.get_support()])
#generate predictions
predictions = backtest(full, rr, predictors)
#define accuracy score
print(accuracy_score(predictions["actual"], predictions["prediction"]))
#63%
#to get better accuracy try a different model instead of ridge regression, change numbers like features