import time
import subprocess
import os

def take_10_second_video(output_path):
    """
    Records a 10-second video using the Raspberry Pi camera.

    Args:
        output_path (str): Path to save the recorded video.
    """
    record_command = f"libcamera-vid -t 10000 --codec h264 -o {output_path} --width 1280 --height 900"
    subprocess.run(record_command, shell=True)

def send_video_to_central(video_path, remote_ip, remote_user, remote_path):
    """
    Sends the recorded video to the central PC.

    Args:
        video_path (str): Local path of the video on the Raspberry Pi.
        remote_ip (str): IP address of the central PC.
        remote_user (str): Username on the central PC.
        remote_path (str): Destination path on the central PC.
    """
    scp_command = f"scp {video_path} {remote_user}@{remote_ip}:{remote_path}"
    subprocess.run(scp_command, shell=True)

def trigger_processing_on_central(remote_ip, remote_user, remote_script, remote_video_path):
    """
    Triggers the central PC to process the uploaded video.

    Args:
        remote_ip (str): IP address of the central PC.
        remote_user (str): Username on the central PC.
        remote_script (str): Path to the script on the central PC.
        remote_video_path (str): Path to the uploaded video on the central PC.
    """
    ssh_command = f"ssh {remote_user}@{remote_ip} 'python3 {remote_script} {remote_video_path}'"
    subprocess.run(ssh_command, shell=True)

def verify_video_transfer(remote_ip, remote_user, remote_video_path):
    """
    Checks if the video exists on the central PC.

    Args:
        remote_ip (str): IP address of the central PC.
        remote_user (str): Username on the central PC.
        remote_video_path (str): Path to the video on the central PC.

    Returns:
        bool: True if the video exists, False otherwise.
    """
    check_command = f"ssh {remote_user}@{remote_ip} 'test -f {remote_video_path} && echo exists'"
    result = subprocess.run(check_command, shell=True, stdout=subprocess.PIPE)
    return b'exists' in result.stdout

def main():
    video_filename = "video.h264"
    video_path = f"/home/pi/{video_filename}"
    remote_ip = "192.168.1.156"  # Replace with the IP address of the central PC
    remote_user = "your_central_user"  # Replace with the username of the central PC
    remote_path = f"/home/{remote_user}/videos/"  # Replace with the destination folder on the central PC
    remote_script = f"/home/{remote_user}/main.py"  # Replace with the path to main.py on the central PC
    delay = 4  # Delay in seconds before triggering processing
    time_between_loops = 20  # idk if 20 seconds are enough, can adjust as needed.

    # Loop until down
    while True:
        print("Starting a new cycle...")

        # Step 1: Record video
        print("Recording video...")
        take_10_second_video(video_path)

        # Step 2: Send video to central PC
        print("Sending video to central PC...")
        send_video_to_central(video_path, remote_ip, remote_user, remote_path)

        # Step 3: Verify video transfer and wait
        print("Verifying video transfer...")
        remote_video_path = f"{remote_path}{video_filename}"
        time.sleep(delay)  # Ensure enough time for transfer
        if verify_video_transfer(remote_ip, remote_user, remote_video_path):
            print("Video transfer verified.")

            # Step 4: Trigger processing
            print("Triggering processing on central PC...")
            trigger_processing_on_central(remote_ip, remote_user, remote_script, remote_video_path)
        else:
            print("Video transfer failed. Retrying next cycle.")

        print("Cycle complete. Waiting for next cycle...")
        time.sleep(time_between_loops)  # Wait for the next cycle 

if __name__ == "__main__":
    main()
