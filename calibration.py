import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import json
import cv2
import numpy as np
from PIL import Image, ImageTk
from detect_clogs import threshold_video_movement

SCALING_FACTOR = 2

def calibrate_homography():
    cycles = threshold_video_movement('output.mjpeg')
    if not cycles:
        print("No cycles detected.")
        return

    frames = []
    for row, cycle in enumerate(cycles):
        frame = cycle['thresholded_image']
        # convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # update the json with the fiducial coordinates
        with open('calibration_data.json', 'r') as f:
            calibration_data = json.load(f)
            row_name = 'A' if row == 0 else 'B'
            coords = cycle['fiducial_coordinates']
            if 1 in coords:
                calibration_data[row_name]['fiducial_coordinates']["1"] = [int(coords[1].x), int(coords[1].y)]
            if 2 in coords:
                calibration_data[row_name]['fiducial_coordinates']["2"] = [int(coords[2].x), int(coords[2].y)]
        with open('calibration_data.json', 'w') as f:
            #print(calibration_data)
            json.dump(calibration_data, f, indent=3)

    for row, frame in enumerate(frames):
        # Load calibration data
        with open('calibration_data.json', 'r') as f:
            calibration_data = json.load(f)

        def update_homography_display():
            section = section_var.get()
            pts_src = np.array([calibration_data[section]["homography"]], dtype=float)
            for i, circle in enumerate(circles):
                # get the center of the circle (average of the four points)
                x = np.mean(canvas.coords(circle)[0:4:2])
                y = np.mean(canvas.coords(circle)[1:4:2])
                # double the x and y values to match the img scaling
                x *= SCALING_FACTOR
                y *= SCALING_FACTOR
                pts_src[0][i] = [x, y]

            # Update calibration data
            calibration_data[section]["homography"] = pts_src.tolist()[0]
            with open('calibration_data.json', 'w') as f:
                json.dump(calibration_data, f, indent=3)

            # Load the updated calibration data
            with open('calibration_data.json', 'r') as f:
                new_calibration_data = json.load(f)
                pts_src_a = np.array([new_calibration_data['A']["homography"]], dtype=float)
                pts_src_b = np.array([new_calibration_data['B']["homography"]], dtype=float)

            # Update homography displays
            update_homography_image(pts_src_a, section, 'A')
            update_homography_image(pts_src_b, section, 'B')

        def update_homography_image(pts_src, section, row):
            width, height = 400, 300
            pts_dst = np.array([
                [0, 0],  # Top-left
                [width - 1, 0],  # Top-right
                [0, height - 1],  # Bottom-left
                [width - 1, height - 1]  # Bottom-right
            ], dtype=float)
            homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
            warped_image = cv2.warpPerspective(frame, homography_matrix, (width, height))

            # Draw red boxes
            num_nozzles = 16
            nozzle_width = width // num_nozzles
            for i in range(num_nozzles):
                x_start = i * nozzle_width
                x_end = (i + 1) * nozzle_width if (i + 1) < num_nozzles else width
                y_start = int(0.10 * height)
                y_end = int(0.90 * height)
                color = (0, 0, 255)
                cv2.rectangle(warped_image, (x_start, y_start), (x_end, y_end), color, 2)

            img = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            if row == 'A':
                homography_label_A.config(image=img_tk)
                homography_label_A.image = img_tk
            else:
                homography_label_B.config(image=img_tk)
                homography_label_B.image = img_tk

        def on_circle_drag(event):
            item = canvas.find_withtag("current")
            canvas.coords(item, event.x - 5, event.y - 5, event.x + 5, event.y + 5)
            update_homography_display()

        # Create tkinter GUI
        root = tk.Tk()
        root.title("Homography Calibration")

        section_var = tk.StringVar(value='A')
        section_dropdown = ttk.Combobox(root, textvariable=section_var, values=['A', 'B'])
        # set the default value to be the current row
        section_var.set('A' if row == 0 else 'B')
        section_dropdown.pack()

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # scale the image to fit the window - 50%
        img = img.resize((img.width // SCALING_FACTOR, img.height // SCALING_FACTOR))
        img_tk = ImageTk.PhotoImage(img)
        canvas = Canvas(root, width=img.width, height=img.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        circles = []
        homography = calibration_data['A']['homography'] if row == 0 else calibration_data['B']['homography']
        for x, y in homography:
            # match img scaling
            x = x // SCALING_FACTOR
            y = y // SCALING_FACTOR
            circle = canvas.create_oval(x-5, y-5, x+5, y+5, fill='red')
            canvas.tag_bind(circle, '<B1-Motion>', on_circle_drag)
            circles.append(circle)

        label_A = tk.Label(root, text="A")
        label_A.pack(side=tk.LEFT)
        homography_label_A = tk.Label(root)
        homography_label_A.pack(side=tk.LEFT)

        label_B = tk.Label(root, text="B")
        label_B.pack(side=tk.RIGHT)
        homography_label_B = tk.Label(root)
        homography_label_B.pack(side=tk.RIGHT)

        update_homography_display()

        root.mainloop()

# Call the calibration function
calibrate_homography()