import subprocess
import pyipmi
import pyipmi.interfaces
import pyipmi.sensor
import time

from typing import Tuple

# setting fan speeds
#
# https://forums.servethehome.com/index.php?resources/supermicro-x9-x10-x11-fan-speed-control.20/
#
# set fan in peripheral zone to 25%: ipmitool raw 0x30 0x70 0x66 0x01 0x01 0x16
#

SENSOR_TYPE_TEMPERATURE = 1

PERIPHERAL_FAST_FAN_TEMP = 75
PERIPHERAL_SLOW_FAN_TEMP = 65
PERIPHERAL_SUPER_SLOW_FAN_TEMP = 45

ZONE_CPU = 0
ZONE_PERIPHERAL = 1

FAN_MODE_STANDARD = 0
FAN_MODE_FULL = 1
FAN_MODE_OPTIMAL = 2
FAN_MODE_HEAVY_IO = 4


ZONE_ITEMS = {
    "CPU Temp": ZONE_CPU,
    "System Temp": ZONE_CPU,
    "Peripheral Temp": ZONE_PERIPHERAL,
    "CPU_VRM Temp": ZONE_CPU,
    "SOC_VRM Temp": ZONE_CPU,
    "VRMABCD Temp": ZONE_CPU,
    "VRMEFGH Temp": ZONE_CPU,
}


# These have their own fan
IGNORE_TEMPS = {
    # Ryzen
    "SOC_VRM Temp",
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


def get_percent(sdr: pyipmi.sdr.SdrFullSensorRecord, reading: Tuple[int]):
    current_temp = reading[0]
    range = sdr.normal_maximum - sdr.nominal_reading
    assert range > 0
    temp_position = current_temp - sdr.nominal_reading
    if temp_position < 0:
        # below nominal
        return 0.0
    return temp_position / range


def adjust_percent(pct: int):
    pct += 10
    return min(pct, 100)


def manage_temp(ipmi: pyipmi.Ipmi):
    max_temps = {
        ZONE_PERIPHERAL: 0.0,
        ZONE_CPU: 0.0,
    }
    max_percents = {
        ZONE_PERIPHERAL: 0.0,
        ZONE_CPU: 0.0,
    }
    sensors = []
    reservation_id = ipmi.reserve_device_sdr_repository()
    for sdr in ipmi.get_repository_sdr_list(reservation_id):
        if "Temp" not in sdr.device_id_string:
            continue
        sensors.append(sdr)
        if len(sensors) == 1:
            print("")
        try:
            reading = ipmi.get_sensor_reading(sdr.number)
        except pyipmi.CompletionCodeError as e:
            if e.cc == 203:
                # Requested data not present (device isn't there/plugged in)
                continue
            raise
        if sdr.device_id_string.startswith("GPU"):
            zone = ZONE_PERIPHERAL
        else:
            zone = ZONE_ITEMS[sdr.device_id_string]
        temp_in_c = reading[0]
        max_temps[zone] = max(max_temps[zone], temp_in_c)
        if sdr.device_id_string == "SOC_VRM Temp":
            print(sdr)
            continue
        percent_high = get_percent(sdr, reading)
        max_percents[zone] = max(percent_high, max_percents[zone])
        print(f"{sdr.device_id_string}: {temp_in_c} degrees C, percent: {percent_high}")

    set_zone_fan_speed(
        speed_percent=adjust_percent(max_percents[ZONE_PERIPHERAL]),
        zone=ZONE_PERIPHERAL,
    )
    set_zone_fan_speed(
        speed_percent=adjust_percent(max_percents[ZONE_CPU]), 
        zone=ZONE_CPU
    )

    print(
        f"Zone Peripheral: {max_temps[ZONE_PERIPHERAL]} "
        f"degrees C, max percent: {max_percents[ZONE_PERIPHERAL]}%, "
        f"speed percent {adjust_percent(max_percents[ZONE_PERIPHERAL])})"
    )
    print(
        f"Zone CPU: {max_temps[ZONE_CPU]} "
        f"degrees C, max percent: {max_percents[ZONE_CPU]}%, "
        f"speed percent {adjust_percent(max_percents[ZONE_CPU])})"
    )


def main():
    # Start high speed in case we error out, we stay at highest setting (which is noticeable)
    set_zone_fan_speed(speed_percent=100, zone=ZONE_PERIPHERAL)
    set_zone_fan_speed(speed_percent=100, zone=ZONE_CPU)

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

            manage_temp(ipmi=ipmi)

            time.sleep(10)

        except Exception as e:
            print(f"IPMI error: {e}")
        finally:
            if ipmi is not None:
                ipmi.session.close()


def mamage_gpu_fans():
    main()


if __name__ == "__main__":
    main()
