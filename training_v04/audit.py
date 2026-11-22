"""Fail-closed audit of a YOLO object detection dataset (no downloads)."""
from pathlib import Path
from collections import Counter
import hashlib
import yaml
from math import isfinite

EXT={".jpg",".jpeg",".png",".bmp",".webp"}
def _digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def audit_dataset(config_path):
    config=Path(config_path).resolve()
    raw=yaml.safe_load(config.read_text(encoding="utf8"))
    if not isinstance(raw,dict) or not isinstance(raw.get("names"),(list,dict)):
        raise ValueError("Dataset YAML must have a class names list or ID mapping")
    names=raw["names"]
    classes=[int(k) for k in names] if isinstance(names,dict) else list(range(len(names)))
    if sorted(classes)!=list(range(len(classes))):raise ValueError("Nonconsecutive class IDs")
    root=Path(raw.get("path",config.parent))
    if not root.is_absolute():root=(config.parent/root).resolve()
    if not root.is_dir():raise FileNotFoundError(f"Missing dataset root: {root}")
    results={"root":str(root),"splits":{},"errors":[],"warnings":[]}
    hashes={}
    for split in ("train","val","test"):
        folder=raw.get(split)
        if not folder:
            results["errors"].append("Missing split: "+split)
            continue
        images=Path(folder)
        if not images.is_absolute():images=root/images
        if not images.is_dir():
            results["errors"].append(f"Missing image folder: {images}")
            continue
        files=sorted(x for x in images.rglob("*") if x.is_file() and x.suffix.lower() in EXT)
        if not files:results["errors"].append("No images in "+split)
        class_counts=Counter()
        for path in files:
            digest=_digest(path)
            prior=hashes.get(digest)
            if prior and prior[0]!=split:
                results["errors"].append(f"Split leakage {prior[0]} and {split}: {path}")
            else:hashes[digest]=(split,str(path))
            label_root=images.parent.parent/"labels"/images.name
            label_file=label_root/path.relative_to(images).with_suffix(".txt")
            if not label_file.exists():
                results["warnings"].append("Missing label, assumed background: "+str(label_file))
                continue
            for lineno,line in enumerate(label_file.read_text(encoding="utf8").splitlines(),1):
                parts=line.split()
                try:
                    if len(parts)!=5:raise ValueError("Expected 5 columns")
                    id_float=float(parts[0]);cls=int(id_float)
                    if cls!=id_float or cls not in classes:raise ValueError("Invalid class")
                    x,y,w,h=(float(n) for n in parts[1:])
                    if not all(isfinite(z) for z in (x,y,w,h)):raise ValueError("NaN/inf")
                    if not (0<w<=1 and 0<h<=1 and 0<=x<=1 and 0<=y<=1):raise ValueError("Invalid box")
                    if x-w/2< -1e-6 or x+w/2>1+1e-6 or y-h/2< -1e-6 or y+h/2>1+1e-6:
                        raise ValueError("Box outside image")
                    class_counts[cls]+=1
                except (ValueError,OverflowError) as exc:
                    results["errors"].append(f"{label_file}:{lineno}: {exc}")
        results["splits"][split]={"images":len(files),"objects_per_class":dict(class_counts)}
        if not class_counts:results["warnings"].append(f"No annotated objects in {split}")
    results["passed"]=not results["errors"]
    return results
if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",default="outputs/dataset_audit.json")
    a=p.parse_args()
    r=audit_dataset(a.data)
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r["passed"] else 2)
