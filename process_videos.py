import cv2
import os
import glob

def extract_frames_from_video(video_path, output_dir, num_frames=5):
    # Setup video capture
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return
        
    # Calculate intervals
    intervals = [int(total_frames * i / num_frames) for i in range(num_frames)]
    
    video_name = os.path.basename(video_path).split('.')[0]
    
    count = 0
    for frame_idx in intervals:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Save the frame
            output_path = os.path.join(output_dir, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1
            
    cap.release()

def process_directory(directory):
    if not os.path.exists(directory):
        return
        
    print(f"Processing videos in {directory}...")
    mp4_files = glob.glob(os.path.join(directory, "*.mp4"))
    
    for idx, video_file in enumerate(mp4_files):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(mp4_files)} videos...")
        extract_frames_from_video(video_file, directory, num_frames=3)

if __name__ == "__main__":
    print("Extracting frames from uploaded videos...")
    process_directory("dataset/real")
    process_directory("dataset/fake")
    print("Done! You can now run the model training script.")
