"""Train a candidate model only after dataset audit and original weights checks."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--device",default="0")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if not Path(a.weights).is_file():p.error("Checkpoint absent; supply verified trained .pt")
    if a.epochs<1 or a.batch<1:p.error("Invalid parameters")
    from training_v04.audit import audit_dataset
    audit=audit_dataset(a.data)
    if not audit["passed"]:p.error("Dataset failed audit: "+"; ".join(audit["errors"][:4]))
    pth=Path("outputs/train_v04")
    pth.mkdir(parents=True,exist_ok=True)
    manifest={"status":"dry_run" if a.dry_run else "training_requested",
      "timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "dataset":str(Path(a.data).resolve()),"weights":str(Path(a.weights).resolve()),
      "epochs":a.epochs,"imgsz":a.imgsz,"batch":a.batch,"device":a.device}
    (pth/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if a.dry_run:print(json.dumps(manifest,indent=2));return
    from ultralytics import YOLO
    YOLO(str(a.weights)).train(data=str(Path(a.data).resolve()),epochs=a.epochs,imgsz=a.imgsz,
      batch=a.batch,device=a.device,seed=42,deterministic=True,patience=25,cos_lr=True,
      project=str(pth),name="experiment",exist_ok=False)
if __name__=="__main__":main()
