import statistic.calcurator as calc
import pandas as pd
import numpy as np

def extract_feature(data):

    #Kinectic
    print("Kinematic")
    V_xyz = calc.derivation(data)
    velocity  = np.sqrt((V_xyz[:,:,0::3]**2)+ (V_xyz[:,:,1::3]**2) + (V_xyz[:,:,2::3]**2))

    A_xyz =  calc.derivation(V_xyz)
    acceleration  = np.sqrt((A_xyz[:,:,0::3]**2)+ (A_xyz[:,:,1::3]**2) + (A_xyz[:,:,2::3]**2))

    J_xyz =  calc.derivation(A_xyz)
    jerk  = np.sqrt((J_xyz[:,:,0::3]**2)+ (J_xyz[:,:,1::3]**2) + (J_xyz[:,:,2::3]**2))
    
    vertical_movements = calc.calc_vertical(data,"full")

    space = calc.calc_space_effort(data,"full")

    # angle
    bodyangle = calc.calc_body_angle(data)
    np.save("bodyangle.npy",bodyangle)

    angle_arm = bodyangle[:,:,2:8]
    angle_leg = bodyangle[:,:,8:]
    angle_torso = bodyangle[:,:,:2]

    g_angle_range_arm = calc.calc_body_angle_max(bodyangle,"range","arm","full")
    g_angle_range_leg = calc.calc_body_angle_max(bodyangle,"range","leg","full")
    g_angle_range_torso = calc.calc_body_angle_max(bodyangle,"range","torso","full")


    # Space
    distance = calc.calc_distance(data,"full")
    area = calc.calc_area(data,"full")

    # Limb
    limb_contraction = calc.calc_limb_contraction(data,"full")

    shoulder_ratio = calc.calc_shoulder_ratio(data,"full")

    V_xyz = calc.derivation(data)
    velocitiy  = np.sqrt((V_xyz[:,:,0::3]**2)+ (V_xyz[:,:,1::3]**2) + (V_xyz[:,:,2::3]**2))
    g_w = (velocitiy**2)
    g_w = np.where(g_w > 1, np.median(g_w), g_w)
    weight = np.percentile(g_w, 95,axis=1)

    features= [angle_arm,angle_leg,g_angle_range_arm,g_angle_range_leg,g_angle_range_torso,shoulder_ratio,velocity,acceleration,jerk,space,vertical_movements,weight,distance,area]

    return features




def normal_vector(arr):
    hip = arr[0]
    Lshoulder = arr[4]
    Rshoulder = arr[8]

    left_vector = Lshoulder - hip
    right_vector = Rshoulder - hip
    norm = np.cross(left_vector,right_vector)
    norm = norm / np.linalg.norm(norm, 2)
    return norm

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def gluLookAtf2(  eye, center, up_ ):

    forward,side,up = np.zeros(3),np.zeros(3),np.zeros(3)
    matrix2, resultMatrix = np.zeros(16),np.zeros(16)

    forward = eye - center
    forward = normalize(forward);


    side = np.cross(forward, up_);
    side = normalize(side);


    up = np.cross(side, forward);

    matrix2[0] = side[0];
    matrix2[4] = side[1];
    matrix2[8] = side[2];
    matrix2[12] = np.dot(side,eye);

    matrix2[1] = up[0];
    matrix2[5] = up[1];
    matrix2[9] = up[2];
    matrix2[13] = np.dot(up,eye);

    matrix2[2] = forward[0];
    matrix2[6] = forward[1];
    matrix2[10] = forward[2];
    matrix2[14] = np.dot(forward,eye);

    matrix2[3] = matrix2[7] = matrix2[11] = 0.0;
    matrix2[15] = 2.0;

    return matrix2.reshape(4,4)


def calc_symmetry_hand(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    list1 = []
    for arr1 in arr:
        list2 = []
        for arr2 in arr1:
            X = arr2
            ones = np.ones((16,1))
            norm = normal_vector(arr2)
            points = (np.concatenate([X,ones],axis=1) @ gluLookAtf2(norm,np.array([0,0,0]), np.array([0,1,0])))[:,:2]
            left_body = points[6:7]
            right_body = points[9:10]

            hip = points[0:1]

            left_body_d_h  = left_body[:,0] -hip[:,0]
            right_body_d_h = hip[:,0]-right_body[:,0]
            symmetry_up_h = np.abs(left_body_d_h - right_body_d_h)
            symmetry_down_h = np.abs(left_body_d_h) + np.abs(right_body_d_h)
            symmetry_h = symmetry_up_h/symmetry_down_h

            left_body_d_v  = left_body[:,1] -hip[:,1]
            right_body_d_v = right_body[:,1]- hip[:,1]
            symmetry_up_v = np.abs(left_body_d_v - right_body_d_v)
            symmetry_down_v = np.abs(left_body_d_v) + np.abs(right_body_d_v)
            symmetry_v = symmetry_up_v/symmetry_down_v


            symmetry = np.abs(symmetry_h)+np.abs(symmetry_v)
            symmetry = np.abs(symmetry).sum()
            list2.append(symmetry)
        list1.append(list2)
    list1 = np.array(list1)
    if(type == "emotion"):
        return calc_mean_emotion(list1)
    else:
        return list1



def reshaping(arr, type = "tile"):
    arrshape = len(arr.shape)
    if(arrshape ==1):
        arr = np.expand_dims(arr,axis=(1,2))
        arr = np.tile(arr,(1,150,1))
    elif(arrshape ==2):
        if(arr.shape[1] > 75):
            arr = arr.reshape((arr.shape[0],arr.shape[1],1))
            #make 150
            margin = 150 - (arr.shape[1])
            sub_arr = np.zeros((arr.shape[0],margin,arr.shape[2]))
            arr = np.concatenate([sub_arr,arr],axis=1)

        else:
            if(type == "tile"):
                arr = arr.reshape((arr.shape[0],1,arr.shape[1]))
                #make 150
                arr = np.tile(arr,(1,150,1))
            else:
                arr = arr.reshape((arr.shape[0],1,arr.shape[1]))
                margin = 150 - (arr.shape[1])
                sub_arr = np.zeros((arr.shape[0],margin,arr.shape[2]))
                arr = np.concatenate([sub_arr,arr],axis=1)
    elif(arrshape == 3 ):
        margin = 150 - (arr.shape[1])
        sub_arr = np.zeros((arr.shape[0],margin,arr.shape[2]))
        arr = np.concatenate([sub_arr,arr],axis=1)

    return arr


def transform_features(arr):
    arrlist = []
    for e in arr:
        arrlist.append(e.mean())
    return np.array(arrlist)





features_trans_g = []
for e in features_g:
    features_trans_g.append(transform_features(e).reshape(-1,1))
Feature_g = np.concatenate(features_trans_g,axis=1)


np.save("rgb_real_feature.npy",Feature_g)
#np.save("expert_feature_48.npy",Feature_e)

Feature_g.shape
