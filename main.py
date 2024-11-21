from detect_clogs import threshold_video_movement, detect_fiducials, homography, classify_nozzles
import time
#from picamzero import Camera
import requests
import json
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def wait_until_fiducial_in_square(x: float, y: float, width: int, height: int) -> bool:
    """
    Waits until the fiducial is within the square defined by the top left corner (x, y) and the width and height
    of the square. Returns True if the fiducial is within the square, False if 10 seconds pass without the fiducial
    being within the square.
    
    Args:
        x: x coordinate of the top left corner of the square
        y: y coordinate of the top left corner of the square
        width: width of the square
        height: height of the square
        
    Returns:
        True if the fiducial is within the square, False otherwise
    """
    #cam = Camera()
    #start_time = time.time()
    #while time.time() - start_time < 10:
    #    #frame = cam.capture()
    #    fiducial_coordinate = detect_fiducial(frame)
    #    if x < fiducial_coordinate[0] < x + width and y < fiducial_coordinate[1] < y + height:
    #        return True
    #    time.sleep(0.1)
    return False
    
def take_10_second_video(output_path: str) -> None:
    """
    Takes a 10 second video and saves it to the output path.
    
    Args:
        output_path: The path to save the video to
    """
    cam = Camera()
    cam.start_recording(output_path)
    time.sleep(10)
    cam.stop_recording()

def check_nozzles(cycles) -> dict[str, list[bool]]:
    """
    Check if the nozzles are clogged by processing each cycle's accumulated mask.

    Args:
        cycles: A list of cycles, each containing the thresholded image and fiducial coordinate.

    Returns:
        A dictionary containing the nozzle report, with the keys 'front' and 'back' and the values being lists of
        booleans indicating whether the nozzle is clogged or not.
    """
    nozzle_status = {}

    for i, cycle in enumerate(cycles):
        thresholded_image = cycle['thresholded_image']
        fiducial_coordinates = cycle['fiducial_coordinates']

        # Warp using both homographies
        warped_front = homography(fiducial_coordinates, thresholded_image, "A")
        warped_back = homography(fiducial_coordinates, thresholded_image, "B")

        # Classify nozzles for both
        report_front, mean_ratio_front = classify_nozzles(warped_front, section="A")
        report_back, mean_ratio_back = classify_nozzles(warped_back, section="B")

        # Decide which report is valid based on mean white ratio
        if mean_ratio_front < 0.1 and mean_ratio_back < 0.1:
            # Both rows are too dark to be valid
            pass
        elif mean_ratio_front > mean_ratio_back:
            # It's the front row
            print(f"Cycle {i}: Front row has a mean white ratio of {mean_ratio_front}, better than back row's {mean_ratio_back}")
            nozzle_status.update(report_front)
        else:
            # It's the back row
            print(f"Cycle {i}: Back row has a mean white ratio of {mean_ratio_back}, better than front row's {mean_ratio_front}")
            nozzle_status.update(report_back)

    return nozzle_status


def send_webhook(nozzle_report: dict[str,list[bool]], video_path: str, email: str) -> None:
    """
    Send a webhook containing the nozzle report and the video to a Slack channel.
    Also sends the video at "output.mjpeg" to the email specified in the email variable.
    
    Args:
        nozzle_report: The nozzle report
        video_path: The path to the video
        email: The email address to send the video to
    """
    #slack_webhook_url = 'INSERT'
    #slack_data = {
    #    'text': f'Nozzles were detected to be clogged: {json.dumps(nozzle_report)}',
    #}

    #response = requests.post(
    #    slack_webhook_url, data=json.dumps(slack_data),
    #    headers={'Content-Type': 'application/json'}
    #)

    #if response.status_code != 200:
    #    print('Request failed with status', response.status_code)
    #else:
    #    print('Request successful')

    # Send the video to the email
    #msg = MIMEMultipart()
    #msg['From'] = 'clog@magnify.com'
    #msg['To'] = email
    #msg['Subject'] = 'Nozzle Clog Detection Video'
    
    #part = MIMEBase('application', 'octet-stream')
    #with open(video_path, 'rb') as file:
    #    part.set_payload(file.read())
    #encoders.encode_base64(part)
    #part.add_header('Content-Disposition', f'attachment; filename={video_path}')
    #msg.attach(part)

    #with smtplib.SMTP('smtp.freesmtpservers.com', 25) as server:
    #    #server.starttls()
    #    #server.login('clog@magnify.com', "")
    #    server.sendmail('clog@magnify.com', email, msg.as_string())
    #    server.quit()


def main():

    cycles = threshold_video_movement('output.mjpeg')

    # Check if the nozzles are clogged
    nozzle_report = check_nozzles(cycles)
    print(nozzle_report)

    # Send the nozzle report to a webhook along with video if nozzles are clogged
    if any(nozzle_report.get('front', [])) or any(nozzle_report.get('back', [])):
        print("Nozzles are clogged, sending report to webhook.")
        send_webhook(nozzle_report, 'output.mjpeg', "jonathanpotter18@gmail.com")
    else:
        print("Nozzles are not clogged, no action needed.")

if __name__ == '__main__':
    main()