import os
import sys
import subprocess
import pyipmi
import pyipmi.interfaces
import pyipmi.sensor
import time

# setting fan speeds
#
# https://forums.servethehome.com/index.php?resources/supermicro-x9-x10-x11-fan-speed-control.20/
#
# set fan in peripheral zone to 25%: ipmitool raw 0x30 0x70 0x66 0x01 0x01 0x16
#

PERIPHERAL_FAST_FAN_TEMP = 75
PERIPHERAL_SLOW_FAN_TEMP = 65
PERIPHERAL_SUPER_SLOW_FAN_TEMP = 47

ZONE_CPU = 0
ZONE_PERIPHERAL = 1

FAN_MODE_STANDARD = 0
FAN_MODE_FULL = 1
FAN_MODE_OPTIMAL = 2
FAN_MODE_HEAVY_IO = 4

lock_file_path = "/tmp/gpu_fan_control.lock"


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


def manage_temp(ipmi: pyipmi.Ipmi, match_str: str, zone: int, current_mode: str):
    max_temp = 0

    sensors = []
    reservation_id = ipmi.reserve_device_sdr_repository()
    for sdr in ipmi.get_repository_sdr_list(reservation_id):
        if match_str in sdr.device_id_string:
            sensors.append(sdr)
            if len(sensors) == 1:
                print("")
            reading = ipmi.get_sensor_reading(sdr.number)
            temp_in_c = reading[0]
            max_temp = max(max_temp, temp_in_c)
            print(f"{sdr.device_id_string}: {temp_in_c} degrees C")
        else:
            continue

    if max_temp <= PERIPHERAL_SUPER_SLOW_FAN_TEMP:
        if current_mode != "super-slow":
            set_zone_fan_speed(speed_percent=25, zone=zone)
            current_mode = "super-slow"
    elif max_temp <= PERIPHERAL_SLOW_FAN_TEMP:
        if current_mode != "slow":
            set_zone_fan_speed(speed_percent=40, zone=zone)
            current_mode = "slow"
    elif max_temp <= PERIPHERAL_FAST_FAN_TEMP:
        if current_mode != "medium":
            set_zone_fan_speed(speed_percent=60, zone=zone)
            current_mode = "medium"
    else:
        if current_mode != "fast":
            set_zone_fan_speed(speed_percent=100, zone=zone)
            current_mode = "fast"

    print(f"Max {match_str} temp: {max_temp} degrees C current mode is {current_mode})")
    return current_mode


def main():
    set_zone_fan_speed(speed_percent=100, zone=ZONE_PERIPHERAL)
    gpu_mode = "fast"

    while True:
        ipmi = None
        try:
            interface = pyipmi.interfaces.create_interface(
                "ipmitool",
                interface_type="open",
            )
            ipmi = pyipmi.create_connection(interface)
            ipmi.session.establish()
            ipmi.target = pyipmi.Target(ipmb_address=0x20)

            gpu_mode = manage_temp(
                ipmi=ipmi, match_str="GPU", zone=ZONE_PERIPHERAL, current_mode=gpu_mode
            )

            time.sleep(10)

        except Exception as e:
            print(f"IPMI error: {e}")
        finally:
            if ipmi is not None:
                ipmi.session.close()


def mamage_gpu_fans():
    main()


if __name__ == "__main__":
    # Check if the lock file already exists
    if os.path.exists(lock_file_path):
        print(f"Another instance of the script {__file__} is already running.")
        sys.exit(1)

    with open(lock_file_path, "w") as lock_file:
        lock_file.write("Lock")
    try:
        main()
    finally:
        # Remove the lock file when done
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
