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

FAST_FAN_TEMP = 75
SLOW_FAN_TEMP = 65
SUPER_LOW_FAN_TEMP = 45

ZONE_CPU = 0
ZONE_PERIPHERAL = 1

FAN_MODE_STANDARD = 0
FAN_MODE_FULL = 1
FAN_MODE_OPTIMAL = 2
FAN_MODE_HEAVY_IO = 4


def set_zone_fan_speed(speed_percent, zone: int = 1):
    ratio = speed_percent / 100
    #fan_speed = int(ratio * 64)
    fan_speed = int(ratio * 90)
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


def main():
    set_zone_fan_speed(speed_percent=100, zone=ZONE_PERIPHERAL)
    mode = "fast"

    while True:
        ipmi = None
        try:
            interface = pyipmi.interfaces.create_interface(
                "ipmitool", interface_type="open"
            )
            ipmi = pyipmi.create_connection(interface)
            ipmi.session.establish()
            ipmi.target = pyipmi.Target(ipmb_address=0x20)

            max_temp = 0

            gpus = []
            reservation_id = ipmi.reserve_device_sdr_repository()
            for sdr in ipmi.get_repository_sdr_list(reservation_id):
                if (
                    sdr.device_id_string.startswith("GPU")
                    and "Temp" in sdr.device_id_string
                ):
                    gpus.append(sdr)
                    if len(gpus) == 1:
                        print("")
                    reading = ipmi.get_sensor_reading(sdr.number)
                    temp_in_c = reading[0]
                    max_temp = max(max_temp, temp_in_c)
                    print(f"{sdr.device_id_string}: {temp_in_c} degrees C")
                else:
                    continue
            
            if max_temp >= FAST_FAN_TEMP and mode != "fast":
                set_zone_fan_speed(speed_percent=100, zone=ZONE_PERIPHERAL)
                mode = "fast"
            elif max_temp <= SUPER_LOW_FAN_TEMP and mode != "super-slow":
                set_zone_fan_speed(speed_percent=20, zone=ZONE_PERIPHERAL)
                mode = "super-slow"
            elif max_temp <= SLOW_FAN_TEMP and mode != "slow":
                set_zone_fan_speed(speed_percent=50, zone=ZONE_PERIPHERAL)
                mode = "slow"
            print(f"Max GPU temp: {max_temp} degrees C current mode is {mode})")

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
