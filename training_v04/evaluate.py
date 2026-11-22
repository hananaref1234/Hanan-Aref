"""Independent held-out YOLO evaluation; only measured results are written."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
CRITICAL_CLASSES=("pole","bollard","stairs","curb","barrier","tree_trunk")
def release_gate(recalls,minimum=.95):
    missing=[name for name in CRITICAL_CLASSES if name not in recalls]
    failed={name:recalls[name] for name in CRITICAL_CLASSES if name in recalls and recalls[name]<minimum}
    return {"target":minimum,"missing":missing,"below_target":failed,"passed":not missing and not failed}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--weights",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--out",default="outputs/v04_evaluation.json")
    a=p.parse_args()
    if not Path(a.weights).is_file():p.error("Missing model weights")
    from ultralytics import YOLO
    model=YOLO(a.weights)
    m=model.val(data=a.data,split="test",device=a.device,plots=True)
    names=model.names
    per_class={}
    for pos,cls in enumerate(m.box.ap_class_index):
        prec,rec,ap50,ap=m.box.class_result(pos)
        name=names[int(cls)]
        per_class[name]={"precision":float(prec),"recall":float(rec),"ap50":float(ap50),"ap50_95":float(ap)}
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "checkpoint":a.weights,"dataset":a.data,"map50":float(m.box.map50),
      "map50_95":float(m.box.map),"per_class":per_class,
      "research_gate":release_gate({k:v["recall"] for k,v in per_class.items()}),
      "notice":"Uncertified research results; hardware validation is still required"}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
