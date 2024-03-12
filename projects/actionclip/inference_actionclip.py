import mmengine
import os
import time
import numpy as np
import torch
import cv2
from mmaction.utils import register_all_modules
from mmaction.apis import inference_recognizer, init_recognizer

# import sys
# sys.path.append('/home/fm-pc-lt-281/projects/mmaction2/projects')

register_all_modules(True)

data = {
    'shoots a wideshot': [],
    'passes the ball': [],
    'scores a goal': [],
    'saves the ball': []
}

# Path to the main folder containing subfolders
main_folder_path = '/home/fm-pc-lt-281/projects/mmaction2/videos/test_data/'

# Iterate through each subfolder
for action_label in data.keys():
    subfolder_path = os.path.join(main_folder_path, action_label)
    if os.path.exists(subfolder_path):
        # Iterate through files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".mp4"):
                data[action_label].append(filename)

print(data)

# config_path = 'work_dirs/actionclip_vit-base-p32-res224-goal_pass_save/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py'
# checkpoint_path = 'work_dirs/actionclip_vit-base-p32-res224-goal_pass_save/best_acc_item_top1_epoch_4.pth'

# config_path = 'work_dirs/actionclip_goals_saves_passes_wideshots_6s/actionclip_goals_saves_passes_wideshots_6s.py'
# checkpoint_path = 'work_dirs/actionclip_goals_saves_passes_wideshots_6s/epoch_125.pth'

# template = 'The player {}'
# # labels = ['scores a goal', 'neither scores a goal nor saves the ball', 'saves the ball']
# labels = ['scores a goal', 'passes the ball', 'saves the ball']


# # Update the labels, the default is the label list of K400.
# config = mmengine.Config.fromfile(config_path)
# config.model.labels_or_label_file = labels

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = init_recognizer(config=config, checkpoint=checkpoint_path, device=device)

# folder_name = '/home/fm-pc-lt-281/projects/mmaction2/videos/test_data_2/'
# for video_name in os.listdir(folder_name):
#     video_path = folder_name + video_name
#     video_name = os.path.basename(video_path)
#     start_time = time.time()
#     pred_result = inference_recognizer(model, video_path)
#     end_time = time.time()

#     # Calculate inference latency
#     inference_latency = end_time - start_time
#     # print("Inference Latency:", inference_latency, "seconds")

#     label_data = pred_result.pred_scores
#     probs = label_data.item.cpu().numpy()

#     # if probs[1] >= 100:
#     #     max_prob_indices = 1
#     #     labels[max_prob_indices] = labels[1] 
#     #     probab = probs[1]
#     # else:
#     #     max_prob_indices = np.argmax(probs)
#     #     probab = probs[max_prob_indices]
#     max_prob_indices = np.argmax(probs)
#     probab = probs[max_prob_indices]

#     # print(max_prob_indices)
#     # print(probs[max_prob_indices])

#     print(f"Predicted label: {labels[max_prob_indices]}, Conf: {probab:.2%}")



# # ## Check if the predicted label is 'score goal'
# # # if labels[max_prob_indices]:
# # #     # Open the input video
# # #     cap = cv2.VideoCapture(video_path)

# # #     # Get video properties
# # #     frame_width = int(cap.get(3))
# # #     frame_height = int(cap.get(4))
# # #     fps = cap.get(5)

# # #     # Create VideoWriter object
# # #     out = cv2.VideoWriter(f'/home/fm-pc-lt-281/projects/mmaction2/videos/output/pass/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # #     # Read each frame and annotate with the label starting from the 10th frame
# # #     for i in range(int(cap.get(7))):  # Total frames in the video
# # #         ret, frame = cap.read()

# # #         if i >= 50:  # Start annotating from the 10th frame
# # #             # Annotate the label on each frame
# # #             # label_text = f'{labels[max_prob_indices]}: {probs[max_prob_indices]:.2%}'
# # #             label_text = f'{labels[max_prob_indices]}'
# # #             label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
# # #             label_x = frame_width - label_size[0] - 60  # Adjusted to fit in the top right corner
# # #             label_y = 131
# # #             cv2.putText(frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# # #         # Write the frame to the output video
# # #         out.write(frame)

# # #     # Release the video capture and writer objects
# # #     cap.release()
# # #     out.release()

# # #     print("Video annotation and saving complete.")
# # # else:
# # #     print("The predicted label is not 'score goal'. No annotation and saving performed.")
