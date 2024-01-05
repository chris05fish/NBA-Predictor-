from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your data and preprocess it (replace this with your actual data loading logic)
df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)
df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

# Feature scaling (replace this with your actual scaling logic)
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Example: Use RidgeClassifier as the model
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split, n_jobs=1)

# Feature selection: train model to pick best features as it learns
# Don't scale the removed columns because they are not numbers
# Scale columns between 0 and 1 for ridge regression
# Old fit data
# sfs.fit(df[selected_columns], df["target"])
# Get list of columns from feature selector 
# predictors = list(selected_columns[sfs.get_support()])

# Split data up by seasons, start = 2 means it needs at least two seasons to predict 
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        # All data before the test season
        train = data[data["season"] < season]
        # Current season to predict (e.g., test set = 2018, train set = 2016, 2017)
        test = data[data["season"] == season]
        # Train model
        model.fit(train[predictors], train["target"])
        # Predict with model
        preds = model.predict(test[predictors])
        # Convert to pandas to make it easier to work with
        preds = pd.Series(preds, index=test.index)
        # Put together values 
        combined = pd.concat([test["target"], preds], axis=1)
        # Rename column 
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Run model
# predictions = backtest(df, rr, predictors)

# Compare the two and find out percentage right
# print(accuracy_score(predictions["actual"], predictions["prediction"]))
# 55%

# Make model better and more accurate, use baseline accuracy (e.g., home team wins 60% of the time)
# Shows percentage of wins by the home team ~57%
df.groupby(["home"]).apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0])

# Compute rolling averages 
df_rolling = df[list(selected_columns) + ["won", "team", "season"]]
# For every row in df, find previous 10 games and calculate mean for performance 
def find_team_averages(team):
    # Select only numeric columns for rolling mean calculation
    numeric_cols = team.select_dtypes(include="number").columns
    rolling = team[numeric_cols].rolling(10).mean()
    return rolling
# Avg performance is specific to each team, only use rolling avg from that season 
df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

# Combine rolling columns with regular
# Rename _10 to end of column name for rolling avg
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
df = pd.concat([df, df_rolling], axis=1)
# Drop rows with missing values
df = df.dropna()

# Give info about the next game, (home/away and opponent)
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

# Indicate if the team will be home or away next game
df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# Pull in stats about the opponent
# DataFrame merge command (like inner SQL join)
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])
# Merge opponents' info with our team info

# Set of columns we don't want to pass into our model (all char or dates/not numbers)
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
# Train model on a new dataframe with new information on next games 
sfs.fit(full[selected_columns], full["target"])
# Feature selector gets the best features to fit the model

# List of predictors 
predictors = list(selected_columns[sfs.get_support()])
# Generate predictions
predictions = backtest(full, rr, predictors)
# Define accuracy score
print(accuracy_score(predictions["actual"], predictions["prediction"]))
# 63%

# To get better accuracy try a different model instead of ridge regression, change numbers like features

# Define the pre-trained model
pretrained_model = (rr, predictors)

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Use the pre-trained model for predictions
        team = request.form['team']
        season = int(request.form['season'])
        date = int(request.form['date'])  # Assuming date is an integer, adjust if it's a different data type

        # Replace this with your actual logic for fetching opponent stats and creating a row for prediction
        input_data = pd.DataFrame({
            'team': [team],
            'season': [season],
            'date': [date],
            # Add other relevant columns based on your dataset
        })

        # Preprocess the input data (scaling, feature selection, etc.)
        input_data[selected_columns] = scaler.transform(input_data[selected_columns])

        # Make prediction using the pre-trained model
        model, predictors = pretrained_model
        prediction = model.predict(input_data[predictors])[0]

        # Display the prediction (you can modify this based on your UI design)
        result = "Win" if prediction == 1 else "Loss"
        return render_template('index.html', result=result)

    # Render the initial form
    return render_template('index.html', result=None)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)