#!/usr/bin/python3
import os
import sys
import subprocess
import pyipmi
import pyipmi.interfaces
import pyipmi.sensor
import time
import pynvml

# setting fan speeds
#
# https://forums.servethehome.com/index.php?resources/supermicro-x9-x10-x11-fan-speed-control.20/
#
# set fan in peripheral zone to 25%: ipmitool raw 0x30 0x70 0x66 0x01 0x01 0x16
#

PERIPHERAL_FAST_FAN_TEMP = 77
PERIPHERAL_MID_FAN_TEMP = 72
PERIPHERAL_SLOW_FAN_TEMP = 65
PERIPHERAL_SUPER_SLOW_FAN_TEMP = 47

ZONE_CPU = 0
ZONE_PERIPHERAL = 1

FAN_MODE_STANDARD = 0
FAN_MODE_FULL = 1
FAN_MODE_OPTIMAL = 2
FAN_MODE_HEAVY_IO = 4

lock_file_path = "/tmp/gpu_fan_control.lock"

# GPUs that have their own cooling system
IGNORE_GPUS = {
    "GPU1 Temp",
}


def set_zone_fan_speed(speed_percent, zone: int = 1):
    ratio = speed_percent / 100
    # fan_speed = int(ratio * 64)
    fan_speed = int(ratio * 95)
    raw_command = [0x30, 0x70, 0x66, 0x01, int(zone), int(fan_speed)]
    cmd = [
        "ipmitool",
        "raw",
    ] + [hex(v) for v in raw_command]
    subprocess.check_call(cmd)


def set_fan_mode(fan_mode: int):
    raw_command = [
        0x30,
        0x45,
    ]
    cmd = [
        "ipmitool",
        "raw",
    ] + [hex(v) for v in raw_command]
    subprocess.check_call(cmd)


def manage_temp(match_str: str, zone: int, current_mode: str):
    max_temp = get_max_gpu_temp()

    if max_temp <= PERIPHERAL_SUPER_SLOW_FAN_TEMP:
        if current_mode != "super-slow":
            set_zone_fan_speed(speed_percent=25, zone=zone)
            current_mode = "super-slow"
    elif max_temp <= PERIPHERAL_SLOW_FAN_TEMP:
        if current_mode != "slow":
            set_zone_fan_speed(speed_percent=40, zone=zone)
            current_mode = "slow"
    elif max_temp <= PERIPHERAL_MID_FAN_TEMP:
        if current_mode != "medium":
            set_zone_fan_speed(speed_percent=60, zone=zone)
            current_mode = "medium"
    elif max_temp <= PERIPHERAL_FAST_FAN_TEMP:
        if current_mode != "medium-x":
            set_zone_fan_speed(speed_percent=75, zone=zone)
            current_mode = "medium-x"
    else:
        if current_mode != "fast":
            set_zone_fan_speed(speed_percent=100, zone=zone)
            current_mode = "fast"

    print(f"Max {match_str} temp: {max_temp} degrees C current mode is {current_mode})")
    return current_mode


def get_max_gpu_temp():
    # Initialize NVML
    pynvml.nvmlInit()
    try:
        # Get the number of GPUs
        device_count = pynvml.nvmlDeviceGetCount()

        max_temp = 0

        # Loop through each GPU and get its temperature
        for i in range(device_count):
            # Get handle for the current GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Get the GPU temperature in Celsius
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                if fan_speed:
                    # Has its own fan, so ignore
                    continue
            except pynvml.NVMLError as ex:
                if ex.value != pynvml.NVML_ERROR_NOT_SUPPORTED:
                    raise

            # Print the temperature
            print(f"GPU {i}: {temperature}°C")
            max_temp = max(temperature, max_temp)
    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()

    return max_temp


def open_ipmi():
    interface = pyipmi.interfaces.create_interface(
        "ipmitool",
        interface_type="open",
    )
    ipmi = pyipmi.create_connection(interface)
    ipmi.session.establish()
    ipmi.target = pyipmi.Target(ipmb_address=0x20)
    return ipmi


def close_ipmi(ipmi):
    if ipmi is not None:
        ipmi.session.close()


def main():
    gpu_mode = "fast"

    while True:
        try:
            gpu_mode = manage_temp(
                match_str="GPU", zone=ZONE_PERIPHERAL, current_mode=gpu_mode
            )

            time.sleep(10)

        except Exception as e:
            print(f"IPMI error: {e}")


def mamage_gpu_fans():
    main()


if __name__ == "__main__":
    # Check if the lock file already exists
    # if os.path.exists(lock_file_path):
    #     print(f"Another instance of the script {__file__} is already running.")
    #     sys.exit(1)

    # with open(lock_file_path, "w") as lock_file:
    #     lock_file.write("Lock")
    try:
        main()
    finally:
        # Remove the lock file when done
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
