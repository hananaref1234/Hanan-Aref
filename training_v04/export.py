"""Intermediate ONNX export; not firmware for Grove Vision AI V2."""
import argparse
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",required=True)
    p.add_argument("--imgsz",type=int,default=320)
    a=p.parse_args()
    if not Path(a.weights).is_file():p.error("No verified checkpoint")
    from ultralytics import YOLO
    output=YOLO(a.weights).export(format="onnx",imgsz=a.imgsz,dynamic=False)
    print("Intermediate ONNX artifact:",output)
    print("Grove Vision AI V2 requires separate Himax/Seeed-compatible deployment validation")
if __name__=="__main__":main()
