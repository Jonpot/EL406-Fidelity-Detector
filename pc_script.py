import subprocess
import time
import os

def send_calibration_data_to_pi(pi_ip, pi_user, pi_calib_path, local_calib_path):
    scp_command = f"scp {local_calib_path} {pi_user}@{pi_ip}:{pi_calib_path}"
    subprocess.run(scp_command, shell=True)

def pull_calibration_data_from_pi(pi_ip, pi_user, pi_calib_path, local_calib_path):
    scp_command = f"scp {pi_user}@{pi_ip}:{pi_calib_path} {local_calib_path}"
    subprocess.run(scp_command, shell=True)

def trigger_recording_on_pi(pi_ip, pi_user, pi_script_path):
    ssh_command = f"ssh {pi_user}@{pi_ip} /usr/bin/python3 {pi_script_path}"
    subprocess.run(ssh_command, shell=True)

def pull_video_from_pi(pi_ip, pi_user, pi_video_path, local_video_path):
    scp_command = f"scp {pi_user}@{pi_ip}:{pi_video_path} {local_video_path}"
    subprocess.run(scp_command, shell=True)

def process_video(local_script_path, video_path):
    process_command = f"python {local_script_path} {video_path}"
    subprocess.run(process_command, shell=True)

def main():
    pi_ip = "192.168.1.164"  # Replace with the Raspberry Pi's IP address
    pi_user = "jonat"        # Replace with the Raspberry Pi's username
    pi_script_path = "/home/jonat/Capstone/EL406-Fidelity-Detector/pi_script.py"
    pi_video_path = "/home/jonat/Capstone/EL406-Fidelity-Detector/video.h264"
    local_video_path = "video.h264"
    local_script_path = "main.py"
    pi_calib_path = "/home/jonat/Capstone/EL406-Fidelity-Detector/calibration_data.json"
    local_calib_path = "calibration_data.json"

    # Calibration step
    print("Checking for calibration data...")
    calibration_exists_command = f"ssh {pi_user}@{pi_ip} test -f {pi_calib_path} && echo exists"
    result = subprocess.run(calibration_exists_command, shell=True, stdout=subprocess.PIPE)
    if b'exists' not in result.stdout:
        print("Calibration data not found on Pi. Sending calibration data...")
        if os.path.exists(local_calib_path):
            send_calibration_data_to_pi(pi_ip, pi_user, pi_calib_path, local_calib_path)
        else:
            print("Error: Calibration data not found on PC.")
            return
    else:
        print("Calibration data exists on Pi. Pulling calibration data to verify...")
        pull_calibration_data_from_pi(pi_ip, pi_user, pi_calib_path, local_calib_path)

    # Recording and processing steps
    print("Triggering recording on Raspberry Pi...")
    trigger_recording_on_pi(pi_ip, pi_user, pi_script_path)

    print("Waiting for recording to complete...")
    time.sleep(22)  # Wait for the recording to finish

    print("Pulling video from Raspberry Pi...")
    pull_video_from_pi(pi_ip, pi_user, pi_video_path, local_video_path)

    print("Verifying video transfer...")
    if os.path.exists(local_video_path):
        print("Video transfer verified. Starting processing...")
        process_video(local_script_path, local_video_path)
    else:
        print("Video transfer failed. Retrying next cycle.")

if __name__ == "__main__":
    main()
