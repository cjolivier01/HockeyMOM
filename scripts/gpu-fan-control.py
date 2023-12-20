import os
import subprocess
import pyipmi
import pyipmi.interfaces
import pyipmi.sensor

GPU_FANS = {
    "GPU3 Temp": ["FANA"],
}

# setting fan speeds
#
# https://forums.servethehome.com/index.php?resources/supermicro-x9-x10-x11-fan-speed-control.20/
#
# set fan in peripheral zone to 25%: ipmitool raw 0x30 0x70 0x66 0x01 0x01 0x16
#

ZONE_CPU = 0
ZONE_PERIPHERAL = 1


def set_zone_fan_speed(ipmi: pyipmi.Ipmi, speed_percent, zone: int = 1):
    ratio = speed_percent / 100
    fan_speed = int(ratio * 64)
    raw_command = [0x30, 0x70, 0x66, 0x01, int(zone), int(fan_speed)]
    cmd = [
        "ipmitool",
        "raw",
    ] + [hex(v) for v in raw_command]
    subprocess.check_call(cmd)


def main(
    ipmi_addr: str = "192.168.1.60", user: str = "ADMIN", password: str = "Pal0Alt0"
):
    ipmi = None
    try:
        interface = pyipmi.interfaces.create_interface(
            "ipmitool", interface_type="open"
        )
        ipmi = pyipmi.create_connection(interface)
        ipmi.session.set_auth_type_user(user, password)
        ipmi.session.establish()
        ipmi.target = pyipmi.Target(ipmb_address=0x20)

        for selector in range(1, 6):
            caps = ipmi.get_dcmi_capabilities(selector)
            print("Selector: {} ".format(selector))
            print("  version:  {} ".format(caps.specification_conformence))
            print("  revision: {}".format(caps.parameter_revision))
            print("  data:     {}".format(caps.parameter_data))

        rsp = ipmi.get_power_reading(1)

        print("Power Reading")
        print("  current:   {}".format(rsp.current_power))
        print("  minimum:   {}".format(rsp.minimum_power))
        print("  maximum:   {}".format(rsp.maximum_power))
        print("  average:   {}".format(rsp.average_power))
        print("  timestamp: {}".format(rsp.timestamp))
        print("  period:    {}".format(rsp.period))
        print("  state:     {}".format(rsp.reading_state))

        gpus = []
        fans = []

        reservation_id = ipmi.reserve_device_sdr_repository()
        for sdr in ipmi.get_repository_sdr_list(reservation_id):
            if (
                sdr.device_id_string.startswith("GPU")
                and "Temp" in sdr.device_id_string
            ):
                print(sdr)
                gpus.append(sdr)
            elif "FAN" in sdr.device_id_string:
                print(sdr)
                fans.append(sdr)
            else:
                continue

        set_zone_fan_speed(ipmi, speed_percent=50, zone=ZONE_PERIPHERAL)

    except Exception as e:
        print(f"IPMI error: {e}")
    finally:
        if ipmi is not None:
            ipmi.session.close()

if __name__ == "__main__":
    main()
