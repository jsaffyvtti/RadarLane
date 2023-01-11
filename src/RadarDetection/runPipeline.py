import ProcessData
import FitModel
import GetMetrics
import pandas as pd

def main():
    arguments = ProcessData.getArguments()
    file_name = arguments["f"]
    csv_file_name = ProcessData.getCSV(file_name)
    file = pd.read_csv(csv_file_name)
    objects = ProcessData.getObjectsDf(file)
    transformed_data = FitModel.transformDataRegression(objects)
    metrics = GetMetrics.metricsDataFrame(transformed_data)
    final_df = GetMetrics.getFinalData(metrics)
    
    objects.to_csv(file_name[:-4] + "_objects.csv", index = False)
    transformed_data.to_csv(file_name[:-4] + "_transformed.csv", index = False)
    metrics.to_csv(file_name[:-4] + "_metrics.csv", index = False)
    final_df.to_csv(file_name[:-4] + "_results.csv", index = False)
    
if __name__ == "__main__":
    main()