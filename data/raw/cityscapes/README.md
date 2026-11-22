# Raw Cityscapes

Register at https://www.cityscapes-dataset.com/ after legal review of the non-commercial license.

Expected layout:

```text
leftImg8bit/train|val|test/<city>/*_leftImg8bit.png
gtFine/train|val|test/<city>/*_gtFine_labelIds.png
```

Do not map `vegetation` to `tree_trunk`. Masks are not YOLO boxes.
