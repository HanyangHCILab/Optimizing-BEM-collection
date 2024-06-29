from cv2 import convexHull
import numpy as np
from scipy.spatial import ConvexHull
def calc_std(arr):
    std_list = []
    n = int(arr.shape[0]/140)
    for i in range(0,n):
        temp = arr[i*140:i*140+140]
        std = temp.std(axis = 1)
        std_mean = std.mean()
        std_list.append(std_mean)
    
    return np.array(std_list)

def derivation(arr):
    return(arr[:,1:] - arr[:,:arr.shape[1]-1])

def calc_mean_emotion(arr):
    mean_list = []
    n = int(arr.shape[0]/140)
    for i in range(0,n):
        temp = arr[i*140:i*140+140]
        mean = []
        for j in range(0,7):
            etemp = temp[j::7]
            emean = etemp.mean()
            mean.append(emean)
        mean_list.append(mean)    
    return np.array(mean_list)

def calc_mean_person(arr):
    mean_list = []
    n = int(arr.shape[0]/140)
    for i in range(0,n):
        temp = arr[i*140:i*140+140]
        mean = []
        for j in range(0,7):
            etemp = temp[j::7]
            emean =[]
            for e in etemp:
                emean.append(e.mean())
            mean.append(emean)
        mean_list.append(mean)    
    return np.array(mean_list)

2100,150,18,3
def norm_arr(arr):
    return np.sqrt(arr**2).sum(axis = len(arr.shape)-1)

def calc_space_effort(arr, type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    totalSpace = norm_arr(arr[:,arr.shape[1]-1] - arr[:,0])
    segement = norm_arr(arr[:,1:] - arr[:,:arr.shape[1]-1]).sum(axis=1)
    space = segement/totalSpace
    fulldata = np.nan_to_num(space, nan= 1)
   
    if(type == "emotion"):
        return calc_mean_emotion(fulldata)
    else:
        return fulldata
  


def calc_convexhull(arr):
    hull_data = []
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    for element in arr:
        hull = [] 
        for moment in element:
            hull.append(ConvexHull(moment).volume)
        hull_data.append(hull)
 
 
    return np.array(hull_data)



def calc_volume(arr):
    volume_data = []
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    for element in arr:
        volume = [] 
        for moment in element:
            max_x,max_y,max_z = moment[:,0].max(),moment[:,1].max(),moment[:,2].max()
            min_x,min_y,min_z = moment[:,0].min(),moment[:,1].min(),moment[:,2].min()
            volume.append( (max_x - min_x) * (max_y - min_y) * (max_z - min_z) )
            
        volume_data.append(volume)
  
    return np.array(volume_data)


def calc_distance(arr, type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    distance_data = np.concatenate([arr[:,:,:1,:1],arr[:,:,:1,2:]],axis=3)

    distance = (distance_data[:,1:] - distance_data[:,:distance_data.shape[1]-1])**2
    distance = distance.sum(axis=3)
    distance = np.sqrt(distance)

    fulldata = distance.sum(axis=2)
  
    if(type == "emotion"):
        return calc_mean_emotion(fulldata)
    else:
        return fulldata

def calc_area(arr,type ="emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    area_data = np.concatenate([arr[:,:,1,:1],arr[:,:,1,2:]],axis=2)
    area = []
    for element in area_data:
        area.append(ConvexHull(element).area)
    area = np.array(area)
    fulldata = area
    if(type == "emotion"):
        return calc_mean_emotion(fulldata)
    else:
        return fulldata



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calc_body_angle(arr):
    """
    0 = hip,  1 = spine,  2 = neck, 3 = head, 
    4 = Larm, 5 = Lforearm, 6 = Lhand,  7 = Rarm, 
    8 = Rforearm, 9 = Rhand, 10 = Lupleg, 11 = Lleg
    12= Lfoot, 13 = Rupleg, 14 = Rleg, 15 = Rfoot, 
    """
    bone_pairs = [[0,1,2],[1,2,3],[1,2,4],[2,4,5],[4,5,6],[1,2,7],[2,7,8],[7,8,9],[1,0,10],[1,0,13],[0,10,11],[10,11,12],[0,13,14],[13,14,15]]
    
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)

    newarr =[]
    for arr_trial in arr:
        trial = []
        for element in arr_trial:
            temp = []
            for p1,p2,p3 in bone_pairs:
                v1 = element[p1] - element[p2]
                v2 = element[p3] - element[p2]
                if(np.linalg.norm(v1) != 0 and np.linalg.norm(v2) != 0 ):
                    angles = angle_between(v1,v2)
                else : angles = 0
                temp.append(angles)
            trial.append(temp)
        newarr.append(trial)
    return np.array(newarr)


def calc_body_angle_max(arr,method = "max", body = "total",type = "emotion"):
    # calcurate body angle with maximum or range value
    if(method == "max"):
        arr = arr.mean(axis = 1)
    elif(method == "range"):
        arr = arr.max(axis = 1) - arr.min(axis = 1)
    else:
        print("error")
    
    #determine the body part to return ( total, arm, leg, limbs)
    if(body == "total"):
        arr = arr
    elif(body == "arm"):
        arr = arr[:,2:8]
    elif(body == "leg"):
        arr = arr[:,8:]
        #arr = np.concatenate([arr[:,8:9],arr[:,10:11]])
    elif(body == "limbs"):
        arr = arr[:,2:]
    elif(body == "torso"):
        arr = arr[:,:2]
    
    fulldata = arr
    if(type == "emotion"):
        return calc_mean_emotion(fulldata)
    else:
        return fulldata
    

def calc_acc(arr):
    acc_list = []
    n = int(arr.shape[0]/140)
    acc = arr[:,1:] - arr[:,:arr.shape[1]-1]
 
    for i in range(0,n):
        temp = acc[i*140:i*140+140]
        mean = temp.mean()
        acc_list.append(mean)
    
    return np.array(acc_list)



def calc_vertical(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    arr_y = arr[:,:,:,1]

    vertical_movements = (arr_y[:,1:] - arr_y[:,:arr_y.shape[1]-1])**2
    vertical_movements = np.sqrt(vertical_movements)
    
    if(type == "emotion"):
        return calc_mean_emotion(vertical_movements)
    else:
        return vertical_movements

def calc_limb_contraction(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    
    limb_contraction = np.concatenate([arr[:,:,3:4]-arr[:,:,6:7],arr[:,:,3:4]-arr[:,:,9:10],arr[:,:,3:4]-arr[:,:,12:13],arr[:,:,3:4]-arr[:,:,15:16]],axis=2)
    limb_contraction = limb_contraction**2
    limb_contraction = np.sqrt(limb_contraction.sum(axis=3))

    trials = limb_contraction.mean(axis=1).mean(axis=1)
    if(type == "emotion"):
        return calc_mean_emotion(trials)
    else:
        return limb_contraction

# def calc_symmetry(arr,type = "emotion"):
#     arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
#     arr = np.moveaxis(arr,0,3)
#     arr[:,:,:,2] = 0 # remove z axis(depth)

#     left_body = np.concatenate([arr[:,:,4:8],arr[:,:,12:15]],axis=2)
#     right_body = np.concatenate([arr[:,:,8:12],arr[:,:,15:18]],axis=2)
#     nose = arr[:,:,0:1]
#     left_body = nose - left_body
#     right_body = right_body - nose

#     symmetry = left_body - right_body
#     symmetry = symmetry**2
#     symmetry = np.sqrt(symmetry.sum(axis=3))
#     fulldata = symmetry
#     symmetry = symmetry.sum(axis=2)
#     symmetry = symmetry.sum(axis=1)

#     trials = symmetry
#     if(type == "emotion"):
#         return calc_mean_emotion(trials)
#     else:
#         return fulldata

def calc_shoulder_ratio(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    shoulder = arr[:,:,4] - arr[:,:,7]
    shoulder = shoulder**2
    shoulder = np.sqrt(shoulder.sum(axis=2))

    shoulder = np.where(shoulder ==0, shoulder.mean(), shoulder)

    arr_local = arr - arr[:,:,0:1]
    arr_local = arr_local**2
    arr_local = np.sqrt(arr_local.sum(axis=3))
    shoulder_ratio = arr_local/shoulder.reshape(shoulder.shape[0],shoulder.shape[1],1)
    trials = shoulder_ratio.sum(axis=1).sum(axis=1)
    if(type == "emotion"):
        return calc_mean_emotion(shoulder_ratio)
    else:
        return shoulder_ratio


def calc_symmetry2(arr):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    v1 = arr[:,:,0]-arr[:,:,2]
    v2 = arr[:,:,3]-arr[:,:,2]
    n1 = np.cross(v1,v2)
    n1 = n1.reshape(2100,150,1,3)
    n1 = np.concatenate([n1 for i in range (7)],axis=2)

    left_body = np.concatenate([arr[:,:,4:8],arr[:,:,12:15]],axis=2)
    right_body = np.concatenate([arr[:,:,8:12],arr[:,:,15:18]],axis=2)

    symmetric_v = right_body-left_body

    arr1=[]
    for i in range(0,n1.shape[0]):
        arr2=[]
        for j in range(0,n1.shape[1]):
            arr3=[]
            for k in range(0,n1.shape[2]):
                angles = angle_between(n1[i,j,k],symmetric_v[i,j,k])
                arr3.append(np.sin(angles))
            arr2.append(arr3)
        arr1.append(arr2)
    symmetry = np.array(arr1)
    return calc_mean_emotion(symmetry)


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

def calc_symmetry(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    list1 = []
    for arr1 in arr:
        list2 = []
        for arr2 in arr1:
            X = arr2
            ones = np.ones((18,1))
            norm = normal_vector(arr2)
            points = (np.concatenate([X,ones],axis=1) @ gluLookAtf2(norm,np.array([0,0,0]), np.array([0,1,0])))[:,:2]
            left_body = np.concatenate([points[4:8],points[12:15]],axis=0)
            right_body = np.concatenate([points[8:12],points[15:18]],axis=0)
          
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
        return calc_mean_emotion(list1.sum(axis=1))
    else:
        return list1  

def calc_symmetry_hand(arr,type = "emotion"):
    arr = np.array([arr[:,:,0::3],arr[:,:,1::3],arr[:,:,2::3]])
    arr = np.moveaxis(arr,0,3)
    list1 = []
    for arr1 in arr:
        list2 = []
        for arr2 in arr1:
            X = arr2
            ones = np.ones((18,1))
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