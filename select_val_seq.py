import numpy as np


video_list_path = './target_val_video_list.txt'
with open(video_list_path, 'r') as file:
    lines = [line.rstrip('\n') for line in file]

dataset_path = './yolox/data/datasets/val_seq.npy'
dataset = np.load(dataset_path,allow_pickle=True).tolist()

matching_elelments = []
for element in dataset:
    for line in lines:
        if line in str(element):
            matching_elelments.append(element)
            break

new_dataset_path = './target_val_seq.npy'
np.save(new_dataset_path, np.array(matching_elelments, dtype=object))

loaded_data = np.load(new_dataset_path, allow_pickle=True).tolist()
print("showing elements.\n")
for i, element in enumerate(loaded_data):
    print(f'{element[0]}\n') 
print(f"total number of elements: {i+1}") 

