"""Experimental original-YOLO + sensor fusion loop. Not a mobility-safety controller."""
import argparse
import json
import logging
import time
from pathlib import Path
from li_stick_next.adapters import convert_upstream_detections
from li_stick_next.engine import SafetyEngine, Sample
from li_stick_next.transport import parse_telemetry

def run(config_path:Path,classes_path:Path,port:str|None,baudrate:int):
    from src.utils.config_loader import load_app_config
    from src.camera.factory import create_camera
    from src.vision.detector import ObjectDetector
    cfg=load_app_config(config_path,classes_path)
    camera=create_camera(cfg.camera)
    detector=ObjectDetector(cfg.detector.model_path,cfg.detector.confidence_threshold,cfg.allowed_classes,cfg.detector.image_size)
    serial_handle=None
    if port:
        import serial
        serial_handle=serial.Serial(port,baudrate=baudrate,timeout=0)
    engine=SafetyEngine(); latest=None; received_at=None; last_seq=-1
    try:
        camera.open()
        while True:
            if serial_handle:
                while serial_handle.in_waiting:
                    line=serial_handle.readline()
                    if not line.endswith(b"\n"):
                        logging.warning("Incomplete telemetry line; discarded")
                        break
                    try:
                        item=parse_telemetry(line.strip())
                        if item.seq<=last_seq:continue
                        latest,received_at,last_seq=item,time.monotonic(),item.seq
                    except ValueError as exc:
                        logging.warning("Bad telemetry: %s",exc)
            ok,frame=camera.read()
            if not ok or frame is None:
                logging.warning("Camera read failed; refusing to declare clearance")
                break
            now=time.monotonic()
            detections,inference_ms=detector.detect(frame,timestamp=time.time())
            dets=convert_upstream_detections(detections,frame.shape[1],frame.shape[0])
            fresh=latest is not None and received_at is not None and now-received_at<=0.5
            info=latest if fresh else None
            alerts=engine.process(Sample(timestamp=now,detections=dets,camera_valid=True,
                ultrasonic_cm=info.ultrasonic_cm if info else None,
                ultrasonic_valid=bool(info and info.ultrasonic_valid),
                speed_mps=info.speed_mps if info else None,
                imu_valid=bool(info and info.imu_valid),
                impact_g=info.impact_g if info else None,
                orientation_change_deg=info.orientation_change_deg if info else None,
                vertical_accel_mps2=info.vertical_accel_mps2 if info else None,
                stillness_seconds=info.stillness_seconds if info else None))
            print(json.dumps({"t":now,"inference_ms":round(inference_ms,2),"detections":len(dets),
                "telemetry_fresh":fresh,"fall_phase":engine.phase.value,
                "alerts":[{"severity":a.severity.value,"message":a.message,"direction":a.direction} for a in alerts]}),flush=True)
    finally:
        camera.close()
        if serial_handle:serial_handle.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("config/config_v03_combined.yaml"))
    p.add_argument("--classes",type=Path,default=Path("config/classes_v03_combined.yaml"))
    p.add_argument("--port")
    p.add_argument("--baudrate",type=int,default=115200)
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(a.config,a.classes,a.port,a.baudrate)
