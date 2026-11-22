"""Normalize original Li-Stick detections from pixels to [0,1]."""
from li_stick_next.engine import Detection
def convert_upstream_detections(items,width,height):
    if width<=0 or height<=0:raise ValueError("Invalid frame dimensions")
    output=[]
    for item in items:
        try:
            x1=max(0,min(width,float(item.x1))); x2=max(0,min(width,float(item.x2)))
            y1=max(0,min(height,float(item.y1))); y2=max(0,min(height,float(item.y2)))
            if x1<x2 and y1<y2:
                output.append(Detection(str(item.class_name),float(item.confidence),x1/width,y1/height,x2/width,y2/height))
        except (TypeError,ValueError,AttributeError):continue
    return tuple(output)
