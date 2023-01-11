import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import optimize, stats
import argparse as ap
# from matplotlib import pyplot as plt

def getArguments():
    parser = ap.ArgumentParser()
    parser.add_argument("-f", metavar="<FILE>", required = True)
    parser.add_argument("-m", metavar="<METHOD>")
    args = parser.parse_args()
    return vars(args)

def transformVelocityRow(row, coefficient):
    new_basis_matrix = getOrthonormalBasisMatrix(coefficient)
    row_array = np.array([row["vx"], row["vy"]])
    new_row = np.linalg.solve(new_basis_matrix, row_array)
    return new_row[0], new_row[1]

def transformVelocity(dataframe):
    coefficient = getAverageCoefficient(dataframe)
    transformed_velocities = transformVelocityRow(dataframe, coefficient)
    dataframe["transformed_vx"] = transformed_velocities[0]
    dataframe["transformed_vy"] = transformed_velocities[1]
    return dataframe


def getSpeed(df):
    velocities = np.array([np.array(df["transformed_vx"]), np.array(df["transformed_vy"])])
    speeds = []
    for vel_x, vel_y in velocities.T:
        speed = np.sqrt(vel_x**2 + vel_y**2)
        speeds.append(speed)
    return speeds

def getLinearParams(radar_data):
    #Reshaping X and Y data for propper format for sklearn
    x_values = np.array(radar_data["x"]).reshape(-1,1) 
    y_values = np.array(radar_data["y"]).reshape(-1,1)
    
    linear_regressor = linear_model.LinearRegression(fit_intercept=True)
    linear_regressor.fit(x_values, y_values)
    
    intercept = linear_regressor.intercept_[0]
    coefficient = linear_regressor.coef_[0][0]
    return intercept, coefficient


def getOrthonormalBasisMatrix(coefficient):
    basisx1 = 1 / np.sqrt(1 + coefficient**2)
    basisx2 = coefficient / np.sqrt(1 + coefficient**2)

    basisy1 = -coefficient / np.sqrt(1 + coefficient**2)
    basisy2 = 1 / np.sqrt(1 + coefficient**2)

    basis_matrix = np.array([[basisx1, basisy1],[basisx2, basisy2]])
    return basis_matrix


def transformRowRegression(row, coefficient, intercept):
    scaled_y = row["y"] - intercept
    new_basis_matrix = getOrthonormalBasisMatrix(coefficient)
    scaled_row = np.array([row["x"], scaled_y])
    new_row = np.linalg.solve(new_basis_matrix, scaled_row)
    return new_row[0], new_row[1]

def transformDataRegression(dataframe):
    intercept, coefficient = getLinearParams(dataframe)
    transformed_positions = dataframe.transpose().apply(lambda x: transformRowRegression(x, coefficient,intercept)).transpose()
    dataframe["transformed_x"] = transformed_positions[0]
    dataframe["transformed_y"] = transformed_positions[1]
    # plt.plot(dataframe["transformed_x"], dataframe["transformed_y"],".", alpha = .5)
    # plt.axhline(intercept)
    # plt.show()
    dataframe = transformVelocity(dataframe)
    dataframe["speed"] = getSpeed(dataframe)
    return dataframe

def getAverageCoefficient(dataframe):
    coefficents = []
    linear_regressor = linear_model.LinearRegression()
    for id in dataframe["newId"].unique():
        single_object_data = dataframe[dataframe["newId"] == id] 
        x_s = np.array(single_object_data["x"]).reshape(-1,1)
        y_s = np.array(single_object_data["y"]).reshape(-1,1)
        
        single_object_model = linear_regressor.fit(x_s, y_s)
        coefficents.append(single_object_model.coef_[0][0])
    return np.mean(coefficents)


def transformBasis(row, coefficient):
    position_vector = np.array([row["x"], row["y"]])
    new_basis_matrix = getOrthonormalBasisMatrix(coefficient)
    new_row = np.linalg.solve(new_basis_matrix, np.array(position_vector))
    return new_row[0], new_row[1]


def gaussSum(x, mean1, mean2, sd1, sd2, a, b):
    return a*stats.norm.pdf(x, loc = mean1, scale = sd1) + b*stats.norm.pdf(x, loc = mean2, scale = sd2)


def getAverageY(df):
    averages = []
    count = 0
    for id in df["newId"].unique():
        count+=1
        single_vehicle = df[df["newId"] == id]
        average_y = np.mean(single_vehicle["transformed_y"])
        averages.append(average_y)
    return averages

def fitAverageY(transformed_average_ys, function_to_fit, bins = 75, init_sd = 1, maxfev = 10000, init_params = None):

    if init_params == None:
        mean1 = np.median(transformed_average_ys) - 2
        mean2 = np.median(transformed_average_ys) + 2
        sd = .5
        amplitude = 30
        init_params = [mean1, mean2, sd, sd, amplitude, amplitude]
        
    hist_values = np.histogram(transformed_average_ys,bins = bins)
    y_values = hist_values[0]
    x_values = hist_values[1][:-1] #left values used for x, final right value removed
    gaussian_model = optimize.curve_fit(
        function_to_fit, 
        x_values, 
        y_values, 
        p0 = init_params,
        maxfev = maxfev)
    final_params = gaussian_model[0]
    return final_params

def transformDataGaussian(dataframe, function_to_fit, init_params = None):
    coefficient = getAverageCoefficient(dataframe)
    transformed_positions = dataframe.transpose().apply(lambda x: transformBasis(x, coefficient)).transpose()
    dataframe["transformed_x"] = transformed_positions[0]
    dataframe["transformed_y"] = transformed_positions[1]
    average_ys = getAverageY(dataframe)
    # plt.hist(average_ys, bins=30)
    # t = np.linspace(-10,5,100)
    params = fitAverageY(average_ys, function_to_fit, init_params=init_params)
    # f = [gaussSum(i, *params) for i in t]
    # plt.plot(t, f)
    # plt.show()
    mean1 = params[0]
    mean2 = params[1]
    midpoint = (mean1 + mean2)/2
    # plt.plot(dataframe["transformed_x"], dataframe["transformed_y"],".", alpha = .5)
    dataframe["transformed_y"] = [i - midpoint for i in dataframe["transformed_y"]]
    #plt.plot(dataframe["transformed_x"], dataframe["transformed_y"],".", alpha = .5)
    # plt.axhline(midpoint)
    # plt.show()
    dataframe = transformVelocity(dataframe)
    dataframe["speed"] = getSpeed(dataframe)
    return dataframe


# class RegressionModel:
    
#     def __init__(self):
#         self.placeholder = None
#         self.midpoint = None
        
#     def fit(self, radar_data):
#         x_values = np.array(radar_data["x"]).reshape(-1,1) 
#         y_values = np.array(radar_data["y"]).reshape(-1,1)
    
#         linear_regressor = linear_model.LinearRegression(fit_intercept=True)
#         linear_regressor.fit(x_values, y_values)
    
#         intercept = linear_regressor.intercept_[0]
#         self.midpoint = intercept
        
#     def predictLane(self, position_data):
#         transformed_positions = position_data.transpose().apply(lambda x: transformRowRegression(x, coefficient,intercept)).transpose()
#         transformed_x = transformed_positions[0]
#         transformed_y = transformed_positions[1]
#         predictions = []
#         for y_value in transformed_y:
#             if y_value > self.midpoint:
                
def main():
    arguments = getArguments()
    file_name = arguments["f"]
    data = pd.read_csv(file_name)
    transformed_data = transformDataRegression(data)
    transformed_data.to_csv(file_name[:-4]+"_transformed.csv", index = False)
    
if __name__ == "__main__":
    main()