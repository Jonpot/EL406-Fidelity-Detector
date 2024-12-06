import subprocess
import os

def take_n_second_video(output_path, numseconds):
    record_command = f"libcamera-vid -t {numseconds * 1000} --codec h264 -o {output_path} --width 1280 --height 900"
    try:
        subprocess.run(record_command, shell=True)
        print(f"Video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to record video: {e}") 

def main():
    video_filename = "video.h264"
    video_path = f"/home/jonat/Capstone/EL406-Fidelity-Detector/{video_filename}"
    calib_filename = "calibration_data.json"
    calib_path = f"/home/jonat/Capstone/EL406-Fidelity-Detector/{calib_filename}"

    print("Checking for calibration data...")
    if not os.path.exists(calib_path):
        print("Calibration data not found. First time!")
        take_n_second_video(video_path, 20)

    print("Calibration data found. Proceeding with recording...")
    take_n_second_video(video_path, 20)

if __name__ == "__main__":
    main()
