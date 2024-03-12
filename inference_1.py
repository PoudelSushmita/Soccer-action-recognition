import cv2
import torch
import clip
from projects.actionclip.models.load import init_actionclip
from mmaction.utils import register_all_modules

register_all_modules(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = init_actionclip('ViT-B/32-8', device=device)

video_name = 'test70_trimmed.mp4'

video_anno = dict(filename=video_name, start_index=0)
video = preprocess(video_anno).unsqueeze(0).to(device)

# template = 'The player is {}'
# labels = ['playing chess', 'playing kickball', 'playing cricket', 'playing paintball']
template = 'The player {} the ball'
# labels = ['is playing cricket', 'is playing chess', 'scores no goal']
labels = ['passes', 'shoot', 'touch']
text = clip.tokenize([template.format(label) for label in labels]).to(device)

with torch.no_grad():
    video_features = model.encode_video(video)
    text_features = model.encode_text(text)

video_features /= video_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100 * video_features @ text_features.T).softmax(dim=-1)
probs = similarity.cpu().numpy()

print("Label probs:", probs)  # [[9.995e-01 5.364e-07 6.666e-04]]

# Get the index of the maximum probability for each video
max_prob_indices = torch.argmax(similarity, dim=-1)

# Get the corresponding labels for each video
predicted_labels = [labels[idx] for idx in max_prob_indices]

print(predicted_labels)
print(f"Predicted label: {predicted_labels[0]}, Probability: {probs[0][max_prob_indices.item()]:.2%}")

if probs[0][max_prob_indices.item()] * 100 < 60:
    predicted_labels[0]='no goal'

# Check if the predicted label is 'score goal'
if predicted_labels[0]:
    # Open the input video
    cap = cv2.VideoCapture(video_name)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    # Create VideoWriter object
    out = cv2.VideoWriter('videos/output/pass_6_.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Read each frame and annotate with the label starting from the 10th frame
    for i in range(int(cap.get(7))):  # Total frames in the video
        ret, frame = cap.read()

        if i >= 50:  # Start annotating from the 10th frame
            # Annotate the label on each frame
            cv2.putText(frame, f'{predicted_labels[0]}: {probs[0][max_prob_indices.item()]:.2%}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print("Video annotation and saving complete.")
else:
    print("The predicted label is not 'score goal'. No annotation and saving performed.")