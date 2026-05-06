

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import math
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
import time
import pandas as pd
import re

DRONEX = -1630
DRONEY = -2187.81


def calc_scale_rmse(px, py, gtx, gty):
    dgt = np.column_stack((px, py))

    clustering = DBSCAN(eps=80, min_samples=5).fit(dgt)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #print("estimated num of clusters", n_clusters)



    clustered_maps_x = []
    clustered_maps_y = []
    unique_labels = set(labels)

    for k in unique_labels:
        class_member_mask = (labels == k)

        clustered_maps_x.append(dgt[class_member_mask, 0])
        clustered_maps_y.append(dgt[class_member_mask ,1])


    mean_x = [np.mean(cl) for cl in clustered_maps_x]
    mean_y = [np.mean(cl) for cl in clustered_maps_y]

    x_distt = [abs(mean_x[i] - val) for i,val in enumerate(clustered_maps_x)]
    y_distt = [abs(mean_y[i] - val) for i,val in enumerate(clustered_maps_y)]

    x_dist = [max(i) for i in x_distt]
    y_dist = [max(i) for i in y_distt]


    temp = zip(mean_x, mean_y, x_dist, y_dist)
    mean_x = []
    mean_y = []
    x_dist = []
    y_dist = []
    for mx, my, dx, dy in temp:
        if dx > 500 or dy > 500:
            continue
        else:
            mean_x.append(mx)
            mean_y.append(my)
            x_dist.append(dx)
            y_dist.append(dy)

    ## GT CLUSTERING
    dgt2 = np.column_stack((gtx, gty))

    clustering2 = DBSCAN(eps=80, min_samples=5).fit(dgt2)
    labels2 = clustering2.labels_
    n_clusters2 = len(set(labels2)) - (1 if -1 in labels2 else 0)


    clustered_maps_x2 = []
    clustered_maps_y2 = []
    unique_labels2 = set(labels2)

    for k in unique_labels2:
        class_member_mask = (labels2 == k)

        clustered_maps_x2.append(dgt2[class_member_mask, 0])
        clustered_maps_y2.append(dgt2[class_member_mask ,1])


    mean_x2 = [np.mean(cl) for cl in clustered_maps_x2]
    mean_y2 = [np.mean(cl) for cl in clustered_maps_y2]

    x_distt2 = [abs(mean_x2[i] - val) for i,val in enumerate(clustered_maps_x2)]
    y_distt2 = [abs(mean_y2[i] - val) for i,val in enumerate(clustered_maps_y2)]

    x_dist2 = [max(i) for i in x_distt2]
    y_dist2 = [max(i) for i in y_distt2]


    temp2 = zip(mean_x2, mean_y2, x_dist2, y_dist2)
    mean_x2 = []
    mean_y2 = []
    x_dist2 = []
    y_dist2 = []
    for mx, my, dx, dy in temp2:
        if dx > 500 or dy > 500:
            continue
        else:
            mean_x2.append(mx)
            mean_y2.append(my)
            x_dist2.append(dx)
            y_dist2.append(dy)




    KD = KDTree(np.column_stack((mean_x2, mean_y2)))
    dist, inde = KD.query(np.column_stack((mean_x, mean_y)), workers = -1)

    #(mean_x, mean_y)[i] is closest to (mean_x2, mean_y2)[inde[i]]
    #mean_x, mean_y, x_dist, y_dist give ellipses with widht/height = 2*mean_x/y
    #area, ie, scale of the mapped trees is given by pi * x_dist[i] * y_dist[i]

    scales = [math.pi*(val/100)*(y_dist[i]/100) for i, val in enumerate(x_dist)]

    scales2 = [math.pi*(val/100)*(y_dist2[i]/100) for i, val in enumerate(x_dist2)]


    differences = [abs(scales[i]) - abs(scales2[val]) for i, val in enumerate(inde)]

    rmse_scale = np.sqrt(np.mean(np.power(differences, 2)))

    return rmse_scale, clustered_maps_x, clustered_maps_y, mean_x, mean_y, x_dist, y_dist, differences, scales, n_clusters


def read_scales(file):
    data = []
    with open(file, "r") as f:
        for i in f.readlines():
            data.append(float(i.replace("\ufeff","")))

    return data








def flip_x(points: np.ndarray, origin_x: float = DRONEX) -> np.ndarray:
    """
    Flip 2D points horizontally around the vertical line x = origin_x.

    points: array of shape (N, 2) or (..., 2) with [x, y]
    origin_x: x-coordinate of the vertical mirror line
    returns: same shape, mirrored horizontally
    """
    points = np.asarray(points, dtype=float)
    flipped = points.copy()
    flipped[..., 0] = 2 * origin_x - flipped[..., 0]
    return flipped

def flip_y(points: np.ndarray, origin_y: float = DRONEY) -> np.ndarray:
    """
    Flip 2D points horizontally around the vertical line x = origin_x.

    points: array of shape (N, 2) or (..., 2) with [x, y]
    origin_x: x-coordinate of the vertical mirror line
    returns: same shape, mirrored horizontally
    """
    points = np.asarray(points, dtype=float)
    flipped = points.copy()
    flipped[..., 1] = 2 * origin_y - flipped[..., 1]
    return flipped

def rotate_90_clockwise(points: np.ndarray, k: int = 1, origin=(DRONEX, DRONEY)) -> np.ndarray:
    """
    Rotate 2D points clockwise by k * 90 degrees around a given origin.

    points: array of shape (N, 2) or (..., 2)
    k: number of 90-degree clockwise rotations
    origin: (ox, oy)
    """
    points = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)
    k = k % 4

    shifted = points - origin

    if k == 0:
        rotated = shifted
    elif k == 1:
        # (x, y) -> (y, -x)
        rotated = np.stack([shifted[..., 1], -shifted[..., 0]], axis=-1)
    elif k == 2:
        # (x, y) -> (-x, -y)
        rotated = -shifted
    else:  # k == 3
        # (x, y) -> (-y, x)
        rotated = np.stack([-shifted[..., 1], shifted[..., 0]], axis=-1)

    return rotated + origin

def rotate_around_origin(points: np.ndarray,
                         angle_deg: float,
                         origin=(DRONEX, DRONEY)) -> np.ndarray:
    """
    Rotate 2D points by angle_deg (clockwise) around a given origin.

    points : array-like, shape (N, 2) or (..., 2)
        Input points in global coordinates.
    angle_deg : float
        Rotation angle in degrees, clockwise is positive.
    origin : tuple (ox, oy)
        Rotation origin in the same coordinates as points.

    Returns
    -------
    rotated : np.ndarray, same shape as points
    """
    points = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Clockwise rotation matrix:
    # [ c  s]
    # [-s  c]
    R = np.array([[c,  s],
                  [-s, c]])

    # Translate so origin -> (0, 0), rotate, then translate back
    shifted = points - origin
    rotated = shifted @ R.T
    return rotated + origin


def open_occupancy_data(inp_file):
    with open(inp_file,"r") as f:
        map_time = f.readline()
        map_resolution = f.readline()
        map_width = f.readline()
        map_height = f.readline()
        map_origin = f.readline()
        map_data = f.readline()

    temp = map_time.split("Time")[1]
    sec = int(temp.split(",")[0].strip().replace("(","").replace("sec=",""))
    nansec = int(temp.split(",")[1].strip().replace(")","").replace("nanosec=",""))
    
    resolution = float(map_resolution.split(":")[1].strip())
    width = int(map_width.split(":")[1].strip())
    height = int(map_height.split(":")[1].strip())

    origin_position = map_origin.split("Point")[1].split("orientation")[0].replace(",","").split(" ")
    origin_position_x = float(origin_position[0].replace("(x=","").strip())
    origin_position_y = float(origin_position[1].replace("y=","").strip())
    origin_position_z = float(origin_position[2].replace("z=","").replace(")","").strip())
    origin_position = [origin_position_x, origin_position_y, origin_position_z]

    data = map_data.split("'b', ")[1].replace(")","").replace("[","").replace("]","").split(",")

    dat = []
    for i in data:
        i = i.strip()
        dat.append(int(i))

    return width, height, origin_position_x, origin_position_y, resolution, dat


def global_wrapper(inp):
    return inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7]


def loctoglob(ix, iy, typ, globals):
    DRONEX, DRONEY, POSX, POSY, RESOLUTION, POSX2, POSY2, RESOLUTION2 = global_wrapper(globals)
    #0 0 is equal to -512.53 and -2000.71
    #1 1 would be equal to -512.53 + 10 and -2000.71 + 10
    #i.e, startx + ix*10, starty + iy*10
    if typ == 1:
        globalsx = DRONEX + POSX*100
        globalsy = DRONEY + POSY*100

        pres = RESOLUTION*100
    else:
        globalsx = DRONEX + POSX2*100
        globalsy = DRONEY + POSY2*100
        pres = RESOLUTION2*100
        

    retx = globalsx + ix*pres 
    rety = globalsy + iy*pres

    return (retx, rety)


def find_global_objects(data, globals, typ=1):
    iy, ix = np.where(data > 50)
    values = data[iy, ix]


    xglob, yglob = loctoglob(ix, iy, typ, globals)

    out = np.column_stack((xglob, yglob, values))

    return out

def read_f(file):
    with open(file,"r") as f:
        data = []
        for line in f.readlines():
            temp = line.replace("\n","")
            temp = temp.replace("\ufeff","")
            temp = temp.split(",")[0:2]
            temp = [float(temp[0]), float(temp[1])]
            if temp not in data:
                data.append(temp)

    xvals = [t[0] for t in data]
    yvals = [t[1] for t in data]
    return xvals, yvals, data



def remove_nonmapped_trees(tree_coords, mapped_coords):
    tree_coords = np.array(tree_coords)
    mapped_coords = np.array(mapped_coords)

    KD = KDTree(tree_coords)

    new_trees = KD.query_ball_point(mapped_coords, 200, workers=-1)

    all_indices = np.unique(np.concatenate(new_trees)).astype(int)

    filtered = tree_coords[all_indices]
    return filtered[:, 0], filtered[:, 1]


def metrics(new_tree_coords, mapped_coords, RGBD_gt, detection_range = 100):
    new_tree = [[val, new_tree_coords[1][i]] for i, val in enumerate(new_tree_coords[0])]
    trees = np.array(new_tree)
    mappe = np.array(mapped_coords)
    mappe_gt = np.array(RGBD_gt)
    
    KD = KDTree(mappe_gt)
    KDm = KDTree(mappe)

    dist, ind = KD.query(mappe, workers=-1)
    dist_map, ind_map = KDm.query(trees, workers=-1)
    
    total_hall = 0
    total_und = 0

    hallucinations = 0
    rmse = 0
    rmse_count = 0
    for i, val in enumerate(dist):
        if val <= detection_range:
            rmse += (val/100)**2
            rmse_count += 1
        else:
            hallucinations += 1
        total_hall += 1

    rmse = np.sqrt(rmse/rmse_count)

    undetected = 0
    for i, val in enumerate(dist_map):
        if val >= detection_range:
            undetected += 1
     
    total_und = len(new_tree_coords[0])

    return (rmse, hallucinations, undetected, total_hall, total_und)


def run_trial(map_file, rgbd_file, debug = 0):



    WIDTH, HEIGHT, POSX, POSY, RESOLUTION, data= open_occupancy_data(map_file) #input data to compare to
    WIDTH2, HEIGHT2, POSX2, POSY2, RESOLUTION2, data2= open_occupancy_data(rgbd_file) #RGBD gt for filtering UE gt based on path
    data = np.reshape(data, (HEIGHT,WIDTH)).astype(np.float32)
    data2 = np.reshape(data2, (HEIGHT2,WIDTH2)).astype(np.float32)

    globals = [DRONEX, DRONEY, POSX, POSY, RESOLUTION, POSX2, POSY2, RESOLUTION2]
    

    #xt1, yt1, d2 = read_f(gt_tree_file)

    #d2 = np.array(d2)
    #d2 = rotate_90_clockwise(d2)
    #d2 = flip_x(d2)
    #d2 = rotate_around_origin(d2, rot_val)
    #xt = [t[0] for t in d2]
    #yt = [t[1] for t in d2]

    #xt, yt är trädens x o y koordinater, dvs

    

    global_objects = find_global_objects(data, globals)
    global_objects2 = find_global_objects(data2, globals, typ=2)

    xt = [t[0] for t in global_objects2[:,0:2]]
    yt = [t[1] for t in global_objects2[:,0:2]]
    d2 = global_objects2[:,0:2]

    

    test = [i[0] for i in global_objects]
    tesy = [i[1] for i in global_objects]

    xt, yt = remove_nonmapped_trees(d2, global_objects2[:,0:2])

    rmse_scale, clustered_maps_x, clustered_maps_y, mean_x, mean_y, x_dist, y_dist, differences, scales, n_clusters = calc_scale_rmse(test, tesy, xt, yt)

    rmse, hallucinations, undetected, num_hall, num_und= metrics([xt,yt], global_objects[:,0:2], global_objects2[:,0:2])


    if debug > 0:
        print(f'rmse {rmse} \nhallucinations {hallucinations} / {num_hall}\nundetected {undetected} / {num_und}\nrmse scale error {rmse_scale}')


    if debug > 1:
        fig = plt.figure(figsize = (13,13))
        plt.xlabel("X coordinate * 100")
        plt.ylabel("Y coordinate * 100")
        plt.scatter(xt, yt, color="green")
        plt.scatter(test, tesy, color="blue", alpha=0.05)
        plt.scatter(DRONEX, DRONEY, color="red")
        plt.show()

    ret_dict = {}
    ret_dict["rmse"] = rmse
    ret_dict["rmse_scale"] = rmse_scale
    ret_dict["hallucinations"] = {"num": hallucinations, "total": num_hall, "percent": hallucinations/num_hall}
    ret_dict["undetected"] = {"num":undetected, "total":num_und, "percent": undetected/num_und}
    ret_dict["n_clusters"] = n_clusters

    return ret_dict, map_file


#### PANDAS STUFF

def parse_input_file_name(inp):
    dat = inp.split("_")
    setup = dat[-4]
    mapid = dat[-3]
    smoke_lvl = dat[-2]
    run_trial = dat[-1].replace(".txt","")

    return setup, mapid, smoke_lvl, run_trial

def read_laten_file(inp_fuse, inp_inf):
    dat = []
    with open(inp_fuse, "r") as f:
        for i in f.readlines():
            dat.append(float(i))

    arr = np.array(dat)
    mean1 = np.mean(arr)

    dat = []
    with open(inp_inf, "r") as f:
        for i in f.readlines():
            dat.append(float(i))

    arr = np.array(dat)
    mean2 = np.mean(arr)

    return mean1, mean2

def read_disp_file(inp):
    dat = []
    with open(inp, "r") as f:
        for i in f.readlines():
            i = i.split(",")[0]
            dat.append(int(i))
    
    arr = np.array(dat)
    return int(np.mean(arr))


def result_to_row(res, disp_file=None, latenfile_fusion=None, latenfile_inf=None):
    map_file = res[1]
    setup_, mapid_, smoke_, run_ = parse_input_file_name(map_file)

    disp = None
    lat_fusion = None
    lat_infer = None

    if disp_file is not None:
        disp = read_disp_file(disp_file)
    if latenfile_fusion is not None:
        lat_fusion, lat_infer = read_laten_file(latenfile_fusion, latenfile_inf)

    res = res[0]
    return {
        "setup": setup_,
        "map_id": mapid_,
        "smoke_level": smoke_,
        "trial_index": run_,
        "rmse": res["rmse"],
        "rmse_scale": res["rmse_scale"],
        "hallucination_count": res["hallucinations"]["num"],
        "hallucination_count_total": res["hallucinations"]["total"],
        "hallucination_percent": res["hallucinations"]["percent"],
        "undetected_count": res["undetected"]["num"],
        "undetected_count_total": res["undetected"]["total"],
        "undetected_percent": res["undetected"]["percent"],
        "num_clusters": res["n_clusters"],
        "number_disparity": disp,
        "latency_fusion": lat_fusion,
        "latency_inference": lat_infer
    }

    #usage is essentially
    #rows = []
    #rows.append(result_to_row(result))
    #where result is from run_trial function







##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


# file parsing system begins here

ROOT = "/home/nexsos/MAPS"
ROOT_DISPARITY = os.path.join(ROOT, "disparity")
ROOT_LATENCY = os.path.join(ROOT, "latency_saved")
ROOT_MAPS = os.path.join(ROOT, "save_run_fused_maps")



def find_all_files(root_dir, occupancy=False):
    files = []
    pattern = re.compile(r"map\d{3,}")
    for root, dire, file in os.walk(root_dir):

        folder_name = root


        if not pattern.search(folder_name):
            continue
            

        

        for fil in file:
            if fil.endswith(".txt"):
                if not occupancy:
                    files.append([root, fil])
                else:
                    if "occupancy" in fil:
                        files.append([root, fil])

    return files


files_disparity = find_all_files(ROOT_DISPARITY)
files_latency = find_all_files(ROOT_LATENCY)
files_maps = find_all_files(ROOT_MAPS, occupancy=True)



'''
['/home/nexsos/MAPS/disparity/map2/FUSION/smoke_level_2/disparity', 'disparity_2026_04_22_15_45_19_FUSION_2_2_5.txt']
['/home/nexsos/MAPS/latency_saved/map2/FUSION/smoke_level_2/inference', 'latency_inference_2026_04_22_15_45_19_FUSION_2_2_5.txt']
['/home/nexsos/MAPS/save_run_fused_maps/map2/FUSION/smoke_level_2/occupancy', 'occupancy_map_2026_04_22_15_58_48_FUSION_2_2_6.txt']
'''


def file_key(inp):
    if "disparity" in inp[1]:
        part1 = [f.replace(".txt","") for f in inp[1].split("_")[7:]]
    else:
        part1 = [f.replace(".txt","") for f in inp[1].split("_")[8:]]

    res_dict = {}
    '''
    res_dict["setup"] = part1[0]
    res_dict["mapid"] = part1[1]
    res_dict["smoke_level"] = part1[2]
    res_dict["run"] = part1[3]
    '''
    res_dict["setup"] = part1[0]+"_"+part1[1]
    res_dict["mapid"] = part1[2]
    res_dict["smoke_level"] = part1[3]
    res_dict["run"] = part1[4]

    return res_dict


#rgbd_files = [f for f in files_maps if file_key(f)["setup"] == "RGBD"]
rgbd_files = [f for f in files_maps if file_key(f)["setup"] == "REAL_STEREO"]


def find_matching_rgbd(key):
    for f in rgbd_files:
        k2 = file_key(f)
        # choose matching criteria; at least mapid and smoke_level
        if k2["mapid"] == key["mapid"]:
            return f
    return None


file_bunches = []
for file_map in files_maps:
    key = file_key(file_map)

    if key["setup"] == "REAL_STEREO":#if key["setup"] == "RGBD":
        continue

    temp = {}
    temp["occupancy"] = file_map
    temp["setup"] = key["setup"]

    temp["gt"] = find_matching_rgbd(key)  # now GT can vary per map

    for file_disp in files_disparity:
        if file_key(file_disp) == key:
            temp["disparity"] = file_disp
            break

    for file_laten in files_latency:
        if file_key(file_laten) == key:
            if "inference" in file_laten[0]:
                temp["latency_inference"] = file_laten
            elif "fusion" in file_laten[0]:
                temp["latency_fusion"] = file_laten

    file_bunches.append(temp)

#file_bunches[i] has 4 files (map, latency, disparity and gt) if fusion or 3 if stereo (no latency)


get_fname = lambda x : os.path.join(x[0], x[1])
rows = []
maxlen = len(file_bunches)
curr = 1
print("metric calculator")
for bunch in file_bunches:
    map_file = get_fname(bunch["occupancy"])
    rgbd_file = get_fname(bunch["gt"])
    disparity = None
    if "disparity" in bunch.keys():
        disparity = get_fname(bunch["disparity"])
    laten_inf = None
    if "latency_inference" in bunch.keys():
        laten_inf = get_fname(bunch["latency_inference"])
    laten_fus = None
    if "latency_fusion" in bunch.keys():
        laten_fus = get_fname(bunch["latency_fusion"])

    temp_res = run_trial(map_file, rgbd_file)

    temp_row = result_to_row(temp_res, disp_file = disparity, latenfile_fusion = laten_fus, latenfile_inf = laten_inf)

    rows.append(temp_row)

    bar_len = 25
    progress = curr / maxlen
    filled = int(bar_len * progress)
    bar = "=" * filled + " " * (bar_len - filled)
    
    print(f'\r[{bar}] progress: {curr}/{maxlen}', end="", flush=True)

    curr += 1




df = pd.DataFrame(rows)
df.to_csv("output.csv")







"""
occ = "occupancy_map_2026_04_22_10_33_51_FUSION_2_0_11.txt"
rgbd = "occupancy_map_2026_04_21_18_24_13_RGBD_2_0_0.txt"

run_trial(occ, rgbd, debug=2)


df = pd.read_csv("output.csv")
df = df.fillna(value=0)

df_filter = df[df["setup"] == "FUSION"]
df_filter2 = df[df["setup"] == "STEREO"]


df_sorted = df_filter.sort_values(by="smoke_level")
df_sorted2 = df_filter2.sort_values(by="smoke_level")

plt.figure()
plt.scatter(df_sorted["smoke_level"].to_numpy(), df_sorted["number_disparity"].to_numpy())
plt.scatter(df_sorted2["smoke_level"].to_numpy(), df_sorted2["number_disparity"].to_numpy())
plt.legend(["fusion","stereo"])
plt.show()
"""