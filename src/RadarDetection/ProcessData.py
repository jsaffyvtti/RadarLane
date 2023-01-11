import pandas as pd
from datetime import datetime
from pytz import timezone
import numpy as np
import argparse as ap
from bagpy import bagreader

def getCSV(file):
    bag = bagreader(file)
    radar_data_directory = bag.message_by_topic("/sms/radar_objects")
    print(radar_data_directory)
    return radar_data_directory

def getArguments():
    parser = ap.ArgumentParser()
    parser.add_argument("-f", metavar="<FILE>", required = True)
    parser.add_argument("-b", metavar="<BOUNDRY>")
    parser.add_argument("-n", metavar="<NAME>")
    args = parser.parse_args()
    return vars(args)


def expandObjects(row):
    objects = row["objects"]
    objects = objects.replace("[","").replace("]","")
    objects = objects.replace("\n", ",")
    objects_list = objects.split("point: ,")[1:]
    
    for i in range(0, len(objects_list)):
        objects_list[i] = objects_list[i].split(",")
        # print(objects_list[i])
        for j in range(0,len(objects_list[i])):
    
            objects_list[i][j] = objects_list[i][j].strip()
        del objects_list[i][3]
        for j in range(3,6):
            objects_list[i][j] = "v" + objects_list[i][j]
    for i in range(0,len(objects_list)-1):
        objects_list[i] = objects_list[i][:-1]
    for object in objects_list:
        object.append("timestamp: " + str(row["header.stamp.secs"]))
        object.append("nsecs: " + str(int(row["header.stamp.nsecs"])))
    return objects_list


def listToDict(data_list):
    '''
    This function takes data that is in the form of a list with entries of 'key: value' 
    format and turns this into a dictionary
    '''
    new_dict = {}
    for feature in data_list:
        feature_list = feature.split(": ")
        new_dict[feature_list[0]] = float(feature_list[1])
    return new_dict


def listOfDictsToDictOfLists(list_of_dicts):
    new_dict = {key: [dic[key] for dic in list_of_dicts] for key in list_of_dicts[0]}
    return new_dict


def convertTime(row, tzone):
    '''
    This function takes the timestamp given in /sms/radar-objects and converts it back into the datetime
    format seen in the bag file.

    Parameters
    ----------
    row : dataframe row, function meant to be applied or mapped to dataframe
    tzone : string
        A string of the timezone the time is 

    Returns
    -------
    datetime - string
        Returns a datetime in the format given by the bag file.

    '''
    timestamp = row["timestamp"]
    tzinfo = timezone(tzone)
    row_datetime = datetime.fromtimestamp(timestamp)
    fmt = '%y-%m-%d %H:%M:%S'
    datetime_string = row_datetime.astimezone(tzinfo).strftime(fmt)
    nsecs_string = str(int(row["nsecs"]))[:3]
    timezone_string = tzinfo.tzname(row_datetime)
    return datetime_string+ "." + nsecs_string + " " + timezone_string


def getUnpackedExpansions(expansions):
    unpacked_expansions = []
    for lists in expansions:
        if len(lists) > 1:
            for sublist in lists:
                unpacked_expansions.append(sublist)
        else:
            unpacked_expansions.append(lists[0])
    return unpacked_expansions

 
def resetVehicleIds(dataframe, reset_boundry = 1000):
    id_list = np.array(dataframe["objectId"])
    new_ids = []
    new_id_dict = {}
    id_positions = {}
    count = 0
    for i in range(0, len(id_list)):
        id = id_list[i]
        if not(id in new_id_dict):
            new_id_dict[id] = count + 1
            count += 1
            id_positions[id] = i
            new_ids.append(new_id_dict[id])
        else:
            dist = i - id_positions[id]
            if dist > reset_boundry:
                new_id_dict[id] = count + 1
                count += 1
                id_positions[id] = i
                new_ids.append(new_id_dict[id])
            else:
                new_ids.append(new_id_dict[id])
                id_positions[id] = i
    return new_ids


def removeCloseData(dataframe, cutoff = 30):
    dataframe = dataframe[dataframe["x"] > cutoff]
    return dataframe 

    
def removeDataByVelocity(dataframe, cutoff = -2):
    dataframe = dataframe[dataframe["vx"] < cutoff]
    return dataframe


def getObjectsDf(radar_data):
    expansions = radar_data.transpose().apply(expandObjects)
    unpacked_expansions = getUnpackedExpansions(expansions)
    expansion_dicts_list = list(map(listToDict, unpacked_expansions))
    dict_of_expansions = listOfDictsToDictOfLists(expansion_dicts_list)
    expansions_df = pd.DataFrame(dict_of_expansions)
    expansions_df["datetime"] = expansions_df.transpose().apply(lambda x: convertTime(x, 'EST'))
    expansions_df["newId"] = resetVehicleIds(expansions_df)
    expansions_df = removeCloseData(expansions_df)
    expansions_df = removeDataByVelocity(expansions_df)
    return expansions_df 


def main():
    arguments = getArguments()
    file_name = arguments["f"]
    data_frame = pd.read_csv(file_name)
    expansions_df = getObjectsDf(data_frame)
    expansions_df.to_csv(file_name[:-4] + "_expanded.csv", index = False)
    
if __name__ == "__main__":
    main()
