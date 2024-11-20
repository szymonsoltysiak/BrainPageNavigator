import brainaccess_board as bb
from brainaccess_board.message_queue import BoardControl
from time import sleep
import time
import numpy as np
import tensorflow as tf

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_one_channel_data(rawdata, channel_name="O2"):
    """Extracts a 500-sample segment from a specified EEG channel."""
    data = rawdata.get_data(picks=[channel_name])[0]  # Get data for the specific channel
    return data[-500:]  

def normalize_sample(sample):
    sample = np.array(sample)
    mean_val = sample.mean()
    std_val = sample.std()

    # Avoid division by zero
    if std_val == 0:
        return np.zeros_like(sample)

    normalized_sample = (sample - mean_val) / std_val
    return normalized_sample

# Load the pre-trained model
model_file = "eye_classification_model.h5"
model = tf.keras.models.load_model(model_file)

from selenium import webdriver
import time

# Ustawienia WebDrivera
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Zmienna globalna
current_scroll_position = 0

# Otwórz stronę internetową
url = "https://kcir.pwr.edu.pl/~mucha/"
driver.get(url)
time.sleep(3)

# Pobierz wysokość widoku
viewport_height = driver.execute_script("return window.innerHeight")

def smooth_scroll_to(target_position, duration=1):
    """Smoothly scroll to a target position with a gradual movement."""
    # Get the current scroll position
    current_position = driver.execute_script("return window.scrollY")
    
    # Calculate the total number of steps (small scroll increments)
    steps = int(abs(target_position - current_position) // 10)
    step_size = (target_position - current_position) / steps
    
    for step in range(steps):
        current_position += step_size
        driver.execute_script(f"window.scrollTo(0, {current_position});")
        time.sleep(duration / steps)  # Adjust the duration based on the number of steps

    # Ensure the final position is exactly the target
    driver.execute_script(f"window.scrollTo(0, {target_position});")


def scroll_down():
    global current_scroll_position
    total_height = driver.execute_script("return document.body.scrollHeight")
    if current_scroll_position + viewport_height < total_height:
        # Calculate the target scroll position
        target_position = current_scroll_position + viewport_height
        # Smoothly scroll to the target position
        smooth_scroll_to(target_position)
        current_scroll_position = target_position
        print(f"Scrolled down to {current_scroll_position}")
    else:
        print("Reached bottom of the page")


def scroll_up():
    global current_scroll_position
    if current_scroll_position > 0:
        # Calculate the target scroll position
        target_position = current_scroll_position - viewport_height
        # Smoothly scroll to the target position
        smooth_scroll_to(target_position)
        current_scroll_position = target_position
        print(f"Scrolled up to {current_scroll_position}")
    else:
        print("Reached top of the page")



# Connect to BrainAccess
db, status = bb.db_connect()
if status:
    board_control = BoardControl()
    response = board_control.get_commands()

    board_control.command(response["data"]["stop_recording"])  
    i=0
    pred_classes = []
    while True:
        if i==0:
            board_control.command(response["data"]["start_recording"]) 
        sleep(0.5)  
        device = db.get_mne()
        rawdata = device[next(iter(device))]  
        data = get_one_channel_data(rawdata)
        if len(data) < 500:
            print("Data length less than 500")
            continue
        data_reshaped = data.reshape((1, data.shape[0], 1))  
        pred = model.predict(normalize_sample(data_reshaped))
        pred_class = np.argmax(pred)
        pred_classes.append(pred_class)
        if len(pred_classes) > 3:
            pred_classes.pop(0)

        if 0 in pred_classes:
            pred_class = 0
        elif 1 in pred_classes:
            pred_class = 1
        else:
            pred_class = 2

        if i%2 == 0:
            if pred_class == 0:
                print("eye_down")
                scroll_down()
            elif pred_class == 1:
                print("eye_up")
                scroll_up()
            else:
                print("none")
        if i==0:
            board_control.command(response["data"]["stop_recording"])

        i+=1
        i = i%100

