import brainaccess_board as bb
from brainaccess_board.message_queue import BoardControl
import matplotlib.pyplot as plt
from time import sleep
import json
import numpy as np

def get_one_channel_data(rawdata, channel_name="O2"):
    data = rawdata.get_data(picks=[channel_name])[0]
    return data[:500]

def save_data_json(data, label="none", file_name="eeg_data.json"):
    eeg_entry = {
        "data": data.tolist() if isinstance(data, np.ndarray) else data,
        "label": label
    }

    try:
        try:
            with open(file_name, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(eeg_entry)

        with open(file_name, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Data saved to {file_name}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")


db, status = bb.db_connect()
if status:
    board_control = BoardControl()
    response = board_control.get_commands()

    board_control.command(response["data"]["stop_recording"])
    for i in range(25):
        board_control.command(response["data"]["start_recording"])
        print("start recording")

        sleep(2.1)
        print("stop recording")
        device = db.get_mne()
        rawdata = device[next(iter(device))]

        data = get_one_channel_data(rawdata)
        if len(data) < 500:
            print("Data length less than 500")
            board_control.command(response["data"]["stop_recording"])
            sleep(2) 
            continue
        save_data_json(data)
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(500), data, label='O2', color='b')
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Amplitude (µV)')
        # plt.title('EEG Data: O2 Channel')
        # plt.legend()
        # plt.grid(True)
        # plt.show() 
        sleep(2)   

        board_control.command(response["data"]["stop_recording"])
    