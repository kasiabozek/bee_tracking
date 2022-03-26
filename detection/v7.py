''' Library to import annotations from V7 software into a format that dataset.py expects.
'''
import json
import math
import numpy as np
import os
from utils.func import FR_H, FR_W
from utils import paths

def get_video_name(data):
    '''
    Extract video identfier from v7 data dict.

    Args: 
      data: dict holding v7 json annotations.
    '''
    return data['image']['original_filename'].replace('.mp4','').replace(' ','_')

def process_json(json_file, should_update_crop_spec=False, crop_w=FR_W, crop_h=FR_H, offset_x=1920-FR_W, offset_y=0):
    ''' Loads json, adds cropping specification should_crop is True.

    Args:
      json_file: str full path to json file.
      should_crop: whether frame will be cropped, will write spec for "cropping_box" to json, which 
        the bounding box defining the cropped frame. Default spec is to crop to top right section of image.
      crop_w: width of cropping_box
      crop_h: height of cropping box
      offset_x: x coordinate for top left corner of cropping_box
      offset_y: y coordinate for top left corner of cropping_box
    Returns:
      dict holding json data
    '''

    with open(json_file, 'r') as f:
      data = json.load(f)

    print('Processing json for', get_video_name(data), '...')

    if not should_update_crop_spec:
        return data

    cropping_spec = data.get('cropping_box') # Returns None if not in data dict.
    if cropping_spec is None:
      print("Adding cropping_box field to json annotation...")
    else:
      print("Updating cropping_box field in json annotation...")

    image = data['image']
    data['cropping_box'] = {
        'width': crop_w,
        'height': crop_h,
        # x, y coordinates representing top left corner of cropped box
        'offset': {
            'y': offset_y,
            'x': offset_x
        }
    }
    with open(json_file, 'w') as f:
      json.dump(data, f, indent=2)
    return data

def write_positions(data, pos_dir=paths.POS_DIR, bee_class_name='dancing_bee'):
    '''Compute and write position coordinates & angle from v7 data dict.
    
    Expects V7 annotations to be a polygon or bounding box with directional vector.
    Grabs center of bounding box and offsets if cropping spec is specified.
    Grabs directional vector and converts from [angle from horizontal] to [angle from vertical].
    Writes one .txt file per frame, where each file holds all bee annotations for that frame and each row
    is one bee annotation holding: x center, y center, bee class (0), angle (deg) clockise from the vertical.

    Args:
      data: dict holding annotation data in v7 format.
      video_name: str to identify original video (.txt files written to a <video_name> sub_dir under <pos_dir>).
      pos_dir: str full path of the output directory.
      bee_class_name: str name of bee class in v7 tool.
    '''
    video_name = get_video_name(data)
    if not os.path.exists(os.path.join(pos_dir, video_name)):
      os.makedirs(os.path.join(pos_dir, video_name))

    for video_annotation in data['annotations']:
        if video_annotation['name'] != bee_class_name:
          continue
        for frame, annotation in video_annotation['frames'].items():
            print('frame:', frame)
            a = annotation['directional_vector']['angle']
            a = ((math.degrees(a) + 90) + 360) % 360
            bb = annotation['bounding_box']
            h,w,x,y = bb['h'],bb['w'],bb['x'],bb['y']
            xc,yc = x+h/2,y+w/2
            cropping_spec = data.get('cropping_box') # Returns None if not in data dict.
            if cropping_spec:
                offset = data['cropping_box']['offset']
                xc,yc = xc-offset['x'], yc-offset['y']
            with open(os.path.join(pos_dir, video_name, "%06d.txt" % int(frame)), 'a') as f:
                np.savetxt(f, [[xc,yc,0,a]], fmt='%i', delimiter=',', newline='\n')

######## MAIN FUNCTION ##############

def import_annotations(input_dir, pos_dir=paths.POS_DIR, should_update_crop_spec=False, crop_w=FR_W, crop_h=FR_H, offset_x=1920-FR_W, offset_y=0,v7_bee_class_name='dancing_bee'):
    '''
    Process v7 json files and write position .txt files in format that dataset.py accepts.

    Args:
      input_dir: dir holding json files.
      Rest of args is documented in process_json().
    '''
    files = os.listdir(input_dir)
    for file in files:
      json_file = os.path.join(input_dir,file)
      data = process_json(json_file, should_update_crop_spec, crop_w, crop_h, offset_x, offset_y)
      write_positions(data, pos_dir, v7_bee_class_name)
