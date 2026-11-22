"""Validated input-only ESP32 JSON-line transport; not MCU firmware."""
import json
from dataclasses import dataclass
from math import isfinite
MAX_LINE_BYTES=1024
@dataclass(frozen=True)
class Telemetry:
    seq:int
    ultrasonic_cm:float|None
    ultrasonic_valid:bool
    imu_valid:bool
    speed_mps:float|None=None
    impact_g:float|None=None
    orientation_change_deg:float|None=None
    vertical_accel_mps2:float|None=None
    stillness_seconds:float|None=None

def parse_telemetry(line:bytes)->Telemetry:
    if not isinstance(line,bytes) or len(line)>MAX_LINE_BYTES:raise ValueError("Invalid line")
    try:o=json.loads(line.decode("utf8"))
    except (ValueError,UnicodeError) as e:raise ValueError("Invalid JSON") from e
    if not isinstance(o,dict) or o.get("type")!="telemetry":raise ValueError("Expected telemetry")
    seq=o.get("seq")
    if isinstance(seq,bool) or not isinstance(seq,int) or seq<0:raise ValueError("Invalid seq")
    for key in ("ultrasonic_valid","imu_valid"):
        if not isinstance(o.get(key),bool):raise ValueError("Invalid "+key)
    def num(key,lo,hi):
        v=o.get(key)
        if v is None:return None
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(float(v)) or not lo<=v<=hi:
            raise ValueError("Invalid "+key)
        return float(v)
    return Telemetry(seq,num("ultrasonic_cm",0,1000),o["ultrasonic_valid"],o["imu_valid"],
        num("speed_mps",0,4),num("impact_g",0,40),num("orientation_change_deg",0,180),
        num("vertical_accel_mps2",-100,100),num("stillness_seconds",0,3600))
