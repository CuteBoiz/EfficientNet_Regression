'''
Efficient-net inference on image/video/images folder

author: phatnt
date: May-19-2022
'''
import os
import cv2
import torch
import glob
import sys
import argparse
import numpy as np

sys.path.append('./efficientnet')
from model import EfficientNet

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--softmax", action='store_true', default=False)
    return parser.parse_args()

def softmax(x):
	'''
		Calculate softmax of an array.
		Args:
			x: an array.
		Return:
			Softmax score of input array.
	'''
	assert x is not None, '[ERROR]: input is none!'
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def infer_preprocess(data, image_size):
	if isinstance(data, np.ndarray):
		data = [data]
	elif isinstance(data, list):
		pass		
	for i in range(len(data)):
		image = data[i]
		image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
		image = np.float32(image)
		image = image*(1/255)
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		image = (image - mean) / std
		data[i] = image.transpose((2, 0, 1))
	data = np.asarray(data).astype(np.float32)
	return data

def infer(args):
    '''
        EfficientNet model infer with data(image/video/images folder)
    '''
    assert os.path.isfile(args.weight), "[ERROR] Could not found {}".format(args.weight)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weight, map_location='cpu')
    imgsz = checkpoint['image_size']
    model = EfficientNet.from_name(f'efficientnet-b{checkpoint["arch"]}', num_classes=checkpoint['num_classes'],
                                    image_size=imgsz, in_channels=checkpoint['in_channels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device=device, dtype=torch.float)
    model.set_swish(memory_efficient=False)
    model.eval()

    images = []
    if os.path.isfile(args.data):
        extentions = args.data.split('.')[-1]
        if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
            print(args.data)
            image = cv2.imread(args.data)
            images.append(image)
        elif extentions in ['mp4', 'mov', 'wmv', 'mkv', 'avi', 'flv']:
            cap = cv2.VideoCapture(args.data)
            if (cap.isOpened()== False):
                  assert Exception("[ERROR] Error opening video stream or file")
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    images.append(frame)
                else: 
                    break
    elif os.path.isdir(args.data):
        files = sorted(glob.glob(os.path.join(args.data, '*')))
        for file in files:
            extentions = file.split('.')[-1]
            if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
                print(file)
                image = cv2.imread(file)
                images.append(image)
            else:
                continue
    else:
        raise Exception(f"[ERROR] Could not load data from {args.data}")

    # Batched
    batched_images = []
    range_num = len(images)//args.batch+1 if len(images)%args.batch > 0 else len(images)//args.batch
    for i in range(range_num):
        batched_images.append([])
    count  = 0
    index = 0
    for i in range(len(images)):
        batched_images[index].append(images[i])
        count+=1
        if count == args.batch:
            count = 0
            index += 1

    with torch.no_grad():
        for batch in batched_images:
            batch = infer_preprocess(batch, imgsz)
            batch = torch.from_numpy(batch).to(device=device, dtype=torch.float)
            predictions = model(batch).cpu().numpy()
            for pred in predictions:
                if args.softmax:
                    pred = softmax(pred)
                np.set_printoptions(suppress=True)
                print(np.array(pred, dtype=float))

if __name__ == '__main__':
	args = parser_args()
	infer(args)