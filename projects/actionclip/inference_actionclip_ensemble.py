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
    'shoots a wideshot': [
        '65b8a5c4df794f002a8f7b06.mp4',  
        '65bb559adf794f002a94bc1c.mp4',
        '65b8adecdf794f002a8fa7a0.mp4',  
        '65bb79d7df794f002a94e30f.mp4',
        '65b8b9a9df794f002a8fe7cd.mp4',  
        '65c30fc58631a20040442b3b.mp4',
        '65b8d5b6df794f002a908755.mp4',  
        '65c338168631a20040446356.mp4',
        '65b8e57edf794f002a90f75a.mp4',  
        '65cb64f7911a1e0040f100e9.mp4'
    ],
    'passes the ball': [
        '6554968f772bfd06144a07c8.mp4',
        '655c948574d98c3180806c8f.mp4',
        '6554a577772bfd06144a5beb.mp4',
        '655dd5ae7fa3cc35ce9f2363.mp4',
        '6554cec2772bfd06144b523c.mp4',
        '655eda628c6d584385faade3.mp4',
        '6556240daa1dc14848e2e4b8.mp4',
        '655eeb598c6d584385fae29b.mp4',
        '65570251955808f7d5cc55cd.mp4',
        '655efa848c6d584385fb16be.mp4',
        '655753de955808f7d5cdf3c8.mp4',
        '655f062f8c6d584385fb4e12.mp4',
        '655c591d84f3692cfcfab793.mp4',
        '655f11ea8c6d584385fb7f0c.mp4',
        '655c816074d98c31807f1f53.mp4',
    ],
    'scores a goal': [
        '64f97558398d55a6feefdb86.mp4',
        '654772d219d1a947e8aaf999.mp4',
        '64f9a76a398d55a6feeffb04.mp4',
        '655ede6f8c6d584385fab604.mp4',
        '64fad0dfb199e1aa8cf9fc23.mp4',
        '655f56298c6d584385fc5363.mp4',
        '651505f13d3b288f841f66a8.mp4',
        '65b8ec9bdf794f002a913430.mp4',
        '6516644f454edb97b4fe40e7.mp4',
        '65bb70a4df794f002a94daf1.mp4',
        '651bd957775b05b3c93fe0bf.mp4',
        '65bb89c7df794f002a94f42e.mp4',
        '65278bc52072ea21d17f2d94.mp4',
        '65c0bcdcd753490035e02542.mp4',
        '653a8a67acb5cebdc57da85c.mp4',
    ],
    'saves the ball': [
        '64d0a8c825807e2f5f98495d.mp4',
        '655c80fe74d98c31807f1801.mp4',
        '64f82a57973fdb874d09d6de.mp4',
        '655c992674d98c318080b218.mp4',
        '64fa9d41b199e1aa8cf9a438.mp4',
        '655c9bfd74d98c318080d2be.mp4',
        '651c085c775b05b3c941d7c1.mp4',
        '655cc66d74d98c3180827f61.mp4',
        '6527d1c1e4034b2a76c76378.mp4',
        '655f02568c6d584385fb3952.mp4',
        '654c7e5d9332e87898c189a5.mp4',
        '655f4c548c6d584385fc2f0c.mp4',
        '654c91509332e87898c1a6d0.mp4',
        '658bdc72445fac001ddcbbf4.mp4',
        '655c3fee84f3692cfcf91709.mp4',
    ],
}
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
    video_path = f'/home/fm-pc-lt-281/projects/mmaction2/videos/test_data/{video_name}'

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
