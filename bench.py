import numpy as np
import onnx
import onnxruntime
import cv2
import time
import sys
from tqdm import tqdm
import pdb

model_path = sys.argv[1]
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
#pdb.set_trace()
output_name = session.get_outputs()[0].name
_, _, h, w = session.get_inputs()[0].shape
print(input_name)
print(output_name)
data = np.random.rand(1,3,h,w).astype(np.float32)

warmup = 10
time_array = []
for i in tqdm(range(500)):
    if i < 10:
        continue
    st = time.time()
    result = session.run([output_name], {input_name: data})
    one_iter = time.time() - st
    time_array.append(one_iter)
time_array = np.array(time_array) * 1000
print(time_array.mean(), np.median(time_array), np.std(time_array))