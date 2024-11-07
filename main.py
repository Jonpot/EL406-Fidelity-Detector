from detect_clogs import threshold_video_movement, detect_fiducial
import time
from picamzero import Camera
import requests
import json

def wait_until_fiducial_in_square(x, y, width, height) -> bool:
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
    cam = Camera()
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = cam.capture()
        fiducial_coordinate = detect_fiducial(frame)
        if x < fiducial_coordinate[0] < x + width and y < fiducial_coordinate[1] < y + height:
            return True
        time.sleep(0.1)
    return False
    
def take_10_second_video(output_path: str):
    """
    Takes a 10 second video and saves it to the output path.
    
    Args:
        output_path: The path to save the video to
    """
    cam = Camera()
    cam.start_recording(output_path)
    time.sleep(10)
    cam.stop_recording()

def check_nozzles(front, back, fiducial_coordinate) -> dict[str,list[bool]]:
    """
    Check if the nozzles are clogged by comparing the front and back images.
    
    Args:
        front: The front image of the nozzle
        back: The back image of the nozzle
        fiducial_coordinate: The coordinate of the fiducial in the image
    
    Returns:
        A dictionary containing the nozzle report, with the keys 'front' and 'back' and the values being lists of
        booleans indicating whether the nozzle is clogged or not.
    """

    # unimplemented
    return {'front': [False, False, False], 'back': [False, False, False]}

def send_webhook(nozzle_report: dict[str,list[bool]], video_path: str):
    """
    Send a webhook containing the nozzle report and the video to a Slack channel.
    
    Args:
        nozzle_report: The nozzle report
        video_path: The path to the video
    """

    slack_webhook_url = 'INSERT'
    slack_data = {
        'text': f'Nozzles were detected to be clogged: {json.dumps(nozzle_report)}',
    }

    response = requests.post(
        slack_webhook_url, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        print('Request failed with status', response.status_code)
    else:
        print('Request successful')



def main():
    # Wait until fiducial is within the absolute square it is expected to be in 
    ok = wait_until_fiducial_in_square(0, 0, 0.1, 0.1)
    if not ok:
        print("Fiducial not found in square for 10 seconds, assuming faulty test start.")
        return

    # Take a 10 second video 
    take_10_second_video('output.mjpeg')

    front, back, fiducial_coordinate = threshold_video_movement('output.mjpeg')

    # Check if the nozzles are clogged
    nozzle_report = check_nozzles(front, back, fiducial_coordinate)

    # Send the nozzle report to a webhook along with video if nozzles are clogged
    if any(nozzle_report['front']) or any(nozzle_report['back']):
        print("Nozzles are clogged, sending report to webhook.")
        send_webhook(nozzle_report, 'output.mjpeg')
    else:
        print("Nozzles are not clogged, no action needed.")

if __name__ == '__main__':
    main()