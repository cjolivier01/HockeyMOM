import threading
import time
from tkinter import Button, Entry, Label, Tk

# Shared variables
shared_data = {"lr": 0, "conf": 0, "foob": 0}


# Function to update variables from GUI entries
def update_values():
    try:
        shared_data["lr"] = float(lr_entry.get())
        shared_data["conf"] = float(conf_entry.get())
        shared_data["foob"] = float(foob_entry.get())
    except ValueError:
        print("Please enter valid numbers")


# GUI Thread Function
def gui_thread():
    global lr_entry, conf_entry, foob_entry

    window = Tk()
    window.title("Modify Variables")

    Label(window, text="lr:").grid(row=0, column=0)
    lr_entry = Entry(window)
    lr_entry.grid(row=0, column=1)

    Label(window, text="conf:").grid(row=1, column=0)
    conf_entry = Entry(window)
    conf_entry.grid(row=1, column=1)

    Label(window, text="foob:").grid(row=2, column=0)
    foob_entry = Entry(window)
    foob_entry.grid(row=2, column=1)

    Button(window, text="Update Values", command=update_values).grid(row=3, columnspan=2)

    window.mainloop()


if __name__ == "__main__":
    # Create and start the GUI thread
    thread = threading.Thread(target=gui_thread)
    thread.daemon = True  # Daemonize thread
    thread.start()

    # Main loop
    try:
        while True:
            print(
                f"lr: {shared_data['lr']}, conf: {shared_data['conf']}, foob: {shared_data['foob']}"
            )
            time.sleep(3)
    except KeyboardInterrupt:
        print("Program terminated by user")
