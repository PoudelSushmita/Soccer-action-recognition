import cv2

def trim_video(input_file, output_file, start_time, end_time):
    # Open the video file
    video_capture = cv2.VideoCapture(input_file)

    # Get video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for H.264 codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read and write frames
    frame_number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret or frame_number > end_frame:
            break

        # Check if the current frame is within the specified range
        if start_frame <= frame_number <= end_frame:
            out.write(frame)

        frame_number += 1

    # Release video capture and writer objects
    video_capture.release()
    out.release()

if __name__ == "__main__":
    input_video_file = "655ad2fa9598a620099799201700451277-059339 (1).mp4"
    output_video_file = "test70_trimmed.mp4"
    start_time_seconds = 1025
    end_time_seconds = 1048

    trim_video(input_video_file, output_video_file, start_time_seconds, end_time_seconds)
