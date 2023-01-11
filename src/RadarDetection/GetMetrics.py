import pandas as pd
import numpy as np
import argparse as ap

def getArguments():
    parser = ap.ArgumentParser()
    parser.add_argument("-f", metavar="<FILE>", required = True)
    args = parser.parse_args()
    return vars(args)

def metricsDataFrame(df, changing_lane_threshold = .5):
    average_ys = []
    starting_xs = []
    final_xs = []
    starting_ys = []
    final_ys = []
    change_in_ys = []
    crossed_borders = []
    lanes = []
    starting_lanes = []
    ending_lanes = []
    average_speeds = []
    max_speeds = []
    min_speeds = []
    change_in_speeds = []
    average_vel_xs = []
    average_vel_ys = []
    new_ids = []
    original_ids = []
    initial_secs =[]
    final_secs = []
    initial_nsecs = []
    final_nsecs = []
    initial_datetimes = []
    final_datetimes = []
    change_in_times = []
    vehicles_df = pd.DataFrame()
    for id in df["newId"].unique():
        new_ids.append(id)

        single_vehicle_data = df[df["newId"] == id]
        original_id = single_vehicle_data["objectId"].unique()[0]
        original_ids.append(original_id)

        # print("average")
        average_y =np.mean(np.array(single_vehicle_data["transformed_y"]))
        average_ys.append(average_y)
        # print("start_x")
        starting_x = max(single_vehicle_data["transformed_x"])
        starting_xs.append(starting_x)
        # print("finale_x")
        final_x = min(single_vehicle_data["transformed_x"])
        final_xs.append(final_x)
        # print("start_y")
        starting_x_index = min(single_vehicle_data.index[single_vehicle_data["transformed_x"] == starting_x])
        starting_y = single_vehicle_data.at[starting_x_index, "transformed_y"]
        starting_ys.append(starting_y)

        final_x_index = max(single_vehicle_data.index[single_vehicle_data["transformed_x"] == final_x])
        final_y = single_vehicle_data.at[final_x_index, "transformed_y"]
        final_ys.append(final_y)

        # print("change in y")
        change_in_ys.append(np.abs(final_y - starting_y))
        # print("crossed border")
        if np.sign(starting_y)*np.sign(final_y) == 1:
            crossed_borders.append(0)
        else:
            crossed_borders.append(1)

        lanes.append(np.sign(average_y))
        starting_lanes.append(np.sign(starting_y))
        ending_lanes.append(np.sign(final_y))
        
        average_speeds.append(np.mean(single_vehicle_data["speed"]))
        
        average_vel_xs.append(np.mean(single_vehicle_data["transformed_vx"]))

        average_vel_ys.append(np.mean(single_vehicle_data["transformed_vy"]))

        max_speed = max(single_vehicle_data["speed"])
        max_speeds.append(max_speed)

        min_speed = min(single_vehicle_data["speed"])
        min_speeds.append(min_speed)
        
        change_in_speeds.append(max_speed - min_speed)

        initial_sec = min(single_vehicle_data["timestamp"])
        initial_secs.append(initial_sec)
        final_sec = max(single_vehicle_data["timestamp"])
        final_secs.append(final_sec)

        initial_nsec = single_vehicle_data.at[starting_x_index, "nsecs"]
        initial_nsecs.append(initial_nsec)
        final_nsec = single_vehicle_data.at[final_x_index, "nsecs"]
        final_nsecs.append(final_nsec)

        initial_datetime = single_vehicle_data.at[starting_x_index, "datetime"]
        initial_datetimes.append(initial_datetime)
        final_datetime = single_vehicle_data.at[final_x_index, "datetime"]
        final_datetimes.append(final_datetime)
        
        change_in_time = float(str(int(final_sec)) +"."+ str(int(final_nsec))) - float(str(int(initial_sec)) +"."+ str(int(initial_nsec)))
        change_in_times.append(change_in_time)

        
    vehicles_df["newId"] = new_ids
    vehicles_df["objectId"] = original_ids
    vehicles_df["average_y"] = average_ys
    vehicles_df["initial_x"] = starting_xs
    vehicles_df["final_x"] = final_xs
    vehicles_df["initial_y"] = starting_ys
    vehicles_df["final_y"] = final_ys
    vehicles_df["change_in_y"] = change_in_ys
    vehicles_df["average_velx"] = average_vel_xs
    vehicles_df["average_vely"] = average_vel_ys
    vehicles_df["average_speed"] = average_speeds
    vehicles_df["max_speed"] = max_speeds
    vehicles_df["min_speed"] = min_speeds
    vehicles_df["change_in_speed"] = change_in_speeds
    vehicles_df["predicted_lane_change"] = crossed_borders
    vehicles_df["predicted_lane"] = lanes
    vehicles_df["initial_lane"] = starting_lanes
    vehicles_df["final_lane"] = ending_lanes
    vehicles_df["initial_timestamp"] = initial_sec
    vehicles_df["final_timestamp"] = final_secs
    vehicles_df["initial_nsec"] = initial_nsecs
    vehicles_df["final_nsec"] = final_nsecs
    vehicles_df["initial_datetime"] = initial_datetimes
    vehicles_df["final_datetime"] = final_datetimes
    vehicles_df["change_in_time"] = change_in_times
    return vehicles_df

def getFinalData(metrics):
    final_data = pd.DataFrame()
    final_data["left_lane_count"] = [sum(metrics["predicted_lane"] == -1)]
    final_data["right_lane_count"] = [sum(metrics["predicted_lane"] == 1)]
    final_data["changed_lane_count"] = [sum(metrics["predicted_lane_change"])]
    return final_data

def main():
    arguments = getArguments()
    file_name = arguments["f"]
    data = pd.read_csv(file_name)
    print(data.columns)
    metrics_data = metricsDataFrame(data)
    metrics_data.to_csv(file_name[:-4] +"_metrics.csv", index = False)
    final_data = getFinalData(metrics_data)
    final_data.to_csv(file_name[:-4] +"_final.csv", index = False)
    print(metrics_data)
    print(final_data)
    
if __name__ == "__main__":
    main()
    