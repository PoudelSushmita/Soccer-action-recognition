import os
import sys
import shlex 
import cv2
import glob
import time
import multiprocessing
from multiprocessing import Pool, current_process

class VideoFrameExtractor:
    def __init__(self, video_location, destination_folder):
        self._video_location = video_location
        self._destination_folder = destination_folder
        self._fps = 10  # Default FPS, you can change it using the setter method
        self._average_frame_extraction_time = 0
        self._average_flow_extraction_time  = 0

    @property
    def video_location(self):
        return self._video_location
    
    @property
    def video_location(self):
        return self._video_location

    @property
    def destination_folder(self):
        return self._destination_folder

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, new_fps):
        if new_fps > 0:
            self._fps = new_fps
        else:
            print("FPS must be greater than 0")

    @property
    def average_frame_extraction_time(self):
        return self._average_frame_extraction_time
        
    @property
    def average_flow_extraction_time(self):
        return self._average_flow_extraction_time

    def extract_frames(self):
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)
        
        cap = cv2.VideoCapture(self.video_location)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(cap.get(cv2.CAP_PROP_FPS))
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        total_time = 0

        # Extracting video name from its path
        video_name = os.path.splitext(os.path.basename(self.video_location))[0]

        # Creating a subfolder inside the destination folder
        video_folder = os.path.join(self.destination_folder, video_name)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # it usually helps skipping some frames
            if frame_count % self.fps == 0:
                frame_filename = os.path.join(video_folder, f'frame_{frame_count // self.fps}.jpg')
                cv2.imwrite(frame_filename, frame)

        cap.release()

    def process_video(self):
        start_time = time.time()
        self.extract_frames()
        end_time = time.time()
        extraction_time = end_time - start_time

        self._average_frame_extraction_time = extraction_time
        
    def process_flow(self):
        start_time = time.time()
        self.run_optical_flow()
        end_time = time.time()
        extraction_time = end_time - start_time

        self._average_flow_extraction_time = extraction_time
        
    def run_optical_flow(vid_item, dev_id=0):
        vid_path = vid_item[0]
        vid_id = vid_item[1]
        vid_name = vid_path.split('/')[-1].split('.')[0]

        out_path = 'OpticalFlow/'
        df_path = 'OpticalFlow/data_path/'

        out_full_path = os.path.join(out_path, vid_name)
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass
    
        current = current_process()
        dev_id = (int(current._identity[0]) - 1) % 2
        image_path = '{}/img'.format(out_full_path)
        flow_x_path = '{}/flow_x'.format(out_full_path)
        flow_y_path = '{}/flow_y'.format(out_full_path)

        out_format = 'mp4'
        new_size = [255,255]

        cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
            shlex.quote(vid_path), shlex.quote(flow_x_path), shlex.quote(flow_y_path), shlex.quote(image_path), dev_id, out_format, new_size[0], new_size[1])
    
        os.system(cmd)
        print('{} {} done'.format(vid_id, vid_name))
        sys.stdout.flush()
        return True
        
    def custom_resume(self):
        '''will implement later'''
        return True

def frame_extraction_worker(video_location, destination_folder):
    extractor = VideoFrameExtractor(video_location, destination_folder)
    extractor.process_video()

if __name__ == '__main__':
    video_location     = 'videos/input/' # change accordingly
    destination_folder = 'videos/output/frames_extracted' # change accordingly
    ext = 'mp4'
    vid_list           =  glob.glob(video_location+'/*.' + ext) # if it doesn't work, please use os.listdir

    num_processes      =  4  # put num_process relatively low
    flow_type          = 'tvl1'
    sanity_checking    = False

    processes          = []

    # for _ in range(num_processes):
    #     p = multiprocessing.Process(target=frame_extraction_worker, args=(vid_list, destination_folder))
    #     processes.append(p)
    #     p.start()

    for vid_path in vid_list:
        p = multiprocessing.Process(target=frame_extraction_worker, args=(vid_path, destination_folder))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    extractor = VideoFrameExtractor(video_location, destination_folder)
    print(f'Average Extraction Time: {extractor.average_frame_extraction_time:.2f} seconds')
    
    if sanity_checking:
        print(f'Average Flow Extraction Time: {extractor.average_flow_extraction_time:.2f} seconds')
    else:
        with Pool() as pool:
            pool.map(VideoFrameExtractor.run_optical_flow, zip(vid_list, range(len(vid_list))))

        # Pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))  # not a great method, if needed we'll improve this
