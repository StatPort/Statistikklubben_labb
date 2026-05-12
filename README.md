# Statistikklubben labb - Sales prediction with XGBoost

## Before the lab
- Install [VS Code](https://code.visualstudio.com/download) or any preferred IDE
- Install [Python](https://www.python.org/downloads/)
- Get files

### Get the files
If you are using VS Code, you can do the following:

1. Open the terminal and run:
```bash
    git clone https://github.com/StatPort/Statistikklubben_labb.git
```
2. In the terminal, navigate into the folder:
```bash
    cd Statistikklubben_labb
```
3. In VS Code: **File → Open Folder → select Statistikklubben_labb**

## During the lab (things to experiment with):
- **Evaluation metrics** — try adding RMSE and MAE. Which one is better and more intuitive? 
- **Visualize predictions** — plot predicted vs actual sales values, what do you notice?
- **Tune the model** — try changing `n_estimators`, `max_depth` or `learning_rate` in `XGBRegressor()`, does the model improve? 
- **Feature engineering** — the model uses `Outlet_Establishment_Year` as a raw number, would it make more sense to convert it to outlet age?
- **Overfitting** — the model scores 0.84 on training data but only 0.52 on test data, what does that tell us?
- **Feature selection** — try removing features with low importance scores, does the model still perform well?
- **Feature importance** — what does the feature importance plot tell us, and how does this differ from a traditional regression model?



This lab is based on the following Kaggle example, with some adjustments:
https://www.kaggle.com/code/aishwarya2210/prediction-of-sales-using-xgboost/notebook
