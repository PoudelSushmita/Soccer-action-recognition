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


# config_path_1 = 'work_dirs/actionclip_vit-base-p32-res224-goal_pass_save/actionclip_vit-base-p32-res224-clip-pre_g8xb16_1x1x8_k400-rgb.py'
# checkpoint_path_1 = 'work_dirs/actionclip_vit-base-p32-res224-goal_pass_save/best_acc_item_top1_epoch_4.pth'

# config_path_2 = 'work_dirs/actionclip_vit-base-p32-res224_goal_saves/actionclip_vit-base-p32-res224_goal_saves.py'
# checkpoint_path_2 = 'work_dirs/actionclip_vit-base-p32-res224_goal_saves/best_acc_item_top1_epoch_3.pth'

# config_path_3 = 'work_dirs/actionclip_goal_saves_noaction____1/actionclip_goal_saves_noaction.py'
# checkpoint_path_3 = 'work_dirs/actionclip_goal_saves_noaction____1/best_acc_item_top1_epoch_9.pth'

config_path_1 = 'work_dirs/actionclip_goals_saves_wideshots_6s/actionclip_goals_saves_wideshots_6s.py'
checkpoint_path_1 = 'work_dirs/actionclip_goals_saves_wideshots_6s/epoch_106.pth'

config_path_2 = 'work_dirs/actionclip_goals_saves_passes_wideshots_6s/actionclip_goals_saves_passes_wideshots_6s.py'
checkpoint_path_2 = 'work_dirs/actionclip_goals_saves_passes_wideshots_6s/epoch_125.pth'

template = 'The player {}'
# labels = ['scores a goal', 'neither scores a goal nor saves the ball', 'saves the ball']
# labels = ['scores a goal', 'saves the ball', 'passes the ball', 'is throwing ball', 'is jugling soccer ball', 'is playing kickball']
# labels = ['scores a goal', 'passes the ball', 'saves the ball']
labels = ['scores a goal', 'passes the ball', 'saves the ball', 'shoots a wideshot', 'is juggling soccer ball', 'is playing kickball']


# Update the labels, the default is the label list of K400.
config_1 = mmengine.Config.fromfile(config_path_1)
config_1.model.labels_or_label_file = labels
# config.model.template = template
device = "cuda" if torch.cuda.is_available() else "cpu"
model_1 = init_recognizer(config=config_1, checkpoint=checkpoint_path_1, device=device)

# Update the labels, the default is the label list of K400.
config_2 = mmengine.Config.fromfile(config_path_2)
config_2.model.labels_or_label_file = labels
# config.model.template = template
device = "cuda" if torch.cuda.is_available() else "cpu"
model_2 = init_recognizer(config=config_2, checkpoint=checkpoint_path_2, device=device)

## Update the labels, the default is the label list of K400.
# config_3 = mmengine.Config.fromfile(config_path_3)
# config_3.model.labels_or_label_file = labels
# # config.model.template = template
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_3 = init_recognizer(config=config_3, checkpoint=checkpoint_path_3, device=device)

# Define the data dictionary
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

class_labels = data.keys()
## Define ground truth labels for each video
ground_truth_labels = {}
for category, videos in data.items():
    for video in videos:
        ground_truth_labels[video] = category

# Initialize variables to count true positives, false positives, and false negatives for each class
tp_scores_a_goal, fp_scores_a_goal, fn_scores_a_goal = 0, 0, 0
tp_passes_the_ball, fp_passes_the_ball, fn_passes_the_ball = 0, 0, 0
tp_saves_the_ball, fp_saves_the_ball, fn_saves_the_ball = 0, 0, 0
tp_shoots_a_wideshot, fp_shoots_a_wideshot, fn_shoots_a_wideshot = 0, 0, 0


# # Loop through each video and compare predicted label with ground truth label
for video_name, ground_truth_label in ground_truth_labels.items():
    video_path = os.path.join(os.path.join(main_folder_path, ground_truth_label), video_name)

    max_prob = 0
    predicted_label = None
    # Make predictions with each model and select the one with the highest probability
    for model in [model_1, model_2]:
        pred_result = inference_recognizer(model, video_path)
        label_data = pred_result.pred_scores
        probs = label_data.item.cpu().numpy()
        max_prob_index = np.argmax(probs)
        if probs[max_prob_index] > max_prob:
            max_prob = probs[max_prob_index]
            predicted_label = labels[max_prob_index]

    # pred_result = inference_recognizer(model, video_path)
    # label_data = pred_result.pred_scores
    # probs = label_data.item.cpu().numpy()
    # max_prob_index = np.argmax(probs)
    # predicted_label = labels[max_prob_index]
    # print(predicted_label+ ',  ' + ground_truth_label)

    # Determine true positives, false positives, and false negatives for each class
    for class_label in class_labels:
        class_label_underscored = class_label.replace(' ', '_')
        # print(class_label_underscored)
        if predicted_label == class_label:
            if ground_truth_label == class_label:
                exec(f'tp_{class_label_underscored} += 1')
            else:
                exec(f'fp_{class_label_underscored} += 1')
        else:
            if ground_truth_label == class_label:
                exec(f'fn_{class_label_underscored} += 1')
    # print(tp_scores_a_goal, tp_passes_the_ball, tp_saves_the_ball)
    # print(fp_scores_a_goal, fp_passes_the_ball, fp_saves_the_ball)
    # print(fn_scores_a_goal, fn_passes_the_ball, fn_saves_the_ball)


    # Calculate false negatives
    for class_label in class_labels:
        if predicted_label != class_label and ground_truth_label == class_label:
            exec(f'fn_{class_label_underscored} += 1')

# Calculate precision and recall for each class
precision_goal = tp_scores_a_goal / (tp_scores_a_goal + fp_scores_a_goal)
recall_goal = tp_scores_a_goal / (tp_scores_a_goal + fn_scores_a_goal)

precision_wideshot = tp_shoots_a_wideshot / (tp_shoots_a_wideshot + fp_shoots_a_wideshot)
recall_wideshot = tp_shoots_a_wideshot / (tp_shoots_a_wideshot + fn_shoots_a_wideshot)

precision_pass = tp_passes_the_ball / (tp_passes_the_ball + fp_passes_the_ball)
recall_pass = tp_passes_the_ball / (tp_passes_the_ball + fn_passes_the_ball)

precision_save = tp_saves_the_ball / (tp_saves_the_ball + fp_saves_the_ball)
recall_save = tp_saves_the_ball / (tp_saves_the_ball + fn_saves_the_ball)

# Calculate F1 score for each class
f1_score_goal = 2 * (precision_goal * recall_goal) / (precision_goal + recall_goal)
f1_score_wideshot = 2 * (precision_wideshot * recall_wideshot) / (precision_wideshot + recall_wideshot)
f1_score_pass = 2 * (precision_pass * recall_pass) / (precision_pass + recall_pass)
f1_score_save = 2 * (precision_save * recall_save) / (precision_save + recall_save)

# Print precision and recall for each class
print(f'Precision for "goal": {precision_goal:.2f}, Recall for "goal": {recall_goal:.2f}')
print(f'Precision for "pass": {precision_pass:.2f}, Recall for "pass": {recall_pass:.2f}')
print(f'Precision for "save": {precision_save:.2f}, Recall for "save": {recall_save:.2f}')
print(f'Precision for "wideshot": {precision_wideshot:.2f}, Recall for "wideshot": {recall_wideshot:.2f}')

# Print F1 score for each class
print(f'F1 Score for "goal": {f1_score_goal:.2f}')
print(f'F1 Score for "pass": {f1_score_pass:.2f}')
print(f'F1 Score for "save": {f1_score_save:.2f}')
print(f'F1 Score for "wideshot": {f1_score_wideshot:.2f}')
