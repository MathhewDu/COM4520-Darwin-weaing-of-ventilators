# Team Lima: weaing-of-ventilators
The original data set is not allowed to be public, therefore only provided a sample data set called "sample_output_weaning_data_all_v4". Before use the models, modify dataset that followed the sample dataset's form.

The folder Model contains the model and the Dataloader. The folder Data contains the data pre-process file.

Use the Data/data_preprocess.ipynb to get processed dataset(windows_df_360.csv and windows_df_360_last.csv).

Move the two datasets generated to the Model folder, then run Model/Model.ipynb or Model/Model.py to run the system.

Two additional files:

1. Data/data_analysis.ipynb, contains all of the record for our team's mining to the data
2. Model/ModelAnalysis.ipynb, contains all of the record for our team's deep analysis to the models, and also different types of models in training that we tried.
