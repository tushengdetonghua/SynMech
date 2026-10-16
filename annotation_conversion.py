"""Convert BlenderProc instance masks and metadata to YOLO segmentation labels.

The conversion preserves hole boundaries and connects disconnected visible regions
of the same instance into one YOLO-compatible polygon path.

Example:
    python annotation_conversion.py --scene-root PATH --output-dir PATH \
        --class-map 1:0,2:1,3:2 --verify-ultralytics
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_mask(path):
    # Read uint16 instance IDs without losing label precision.
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Cannot read instance PNG")
    if mask.ndim == 3:
        if not np.all(mask == mask[:, :, :1]):
            raise ValueError("Instance masks must be single-channel ID images")
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError("Instance masks must be 2-D")
    return mask


def save_png(path, array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise OSError("PNG encoding failed")
    encoded.tofile(str(path))


def load_records(path):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    labels = data if isinstance(data, list) else data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list):
        raise ValueError("JSON must be a list or a dict containing labels:list")
    records, duplicates = {}, 0
    for obj in labels:
        if not isinstance(obj, dict) or "instance_id" not in obj or "category_id" not in obj:
            raise ValueError("JSON record is missing instance_id/category_id")
        iid, cid = int(obj["instance_id"]), int(obj["category_id"])
        if iid <= 0:
            continue
        if iid in records:
            duplicates += 1
            if int(records[iid]["category_id"]) != cid:
                raise ValueError(f"instance_id={iid} 对应不同类别；不输出此图片的残缺标签")
        else:
            records[iid] = obj
    return records, duplicates


def nearest_pair(a, b, chunk=256):
    """分块找最近边界像素，避免一次分配 N*M*2 的大数组。"""
    best = (float("inf"), 0, 0)
    for i in range(0, len(a), chunk):
        aa = a[i:i + chunk].astype(np.float64)
        for j in range(0, len(b), chunk):
            d = aa[:, None, :] - b[None, j:j + chunk, :]
            dist = (d * d).sum(axis=2)
            p, q = np.unravel_index(dist.argmin(), dist.shape)
            candidate = (float(dist[p, q]), i + int(p), j + int(q))
            if candidate[0] < best[0]:
                best = candidate
    return best


def compact_exact(points):
    """仅删连续重复点和同方向共线点；保留折返点、非连续重复锚点。

    不能 np.unique(points, axis=0)，也不能对缝合后的路径做 approxPolyDP。
    """
    if len(points) < 3:
        return points
    keep = np.r_[True, np.any(points[1:] != points[:-1], axis=1)]
    points = points[keep]
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return points
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    return points[(cross != 0) | (dot <= 0)]


def encode_instance(binary, allow_subpixel_fallback=True):
    """全部层级轮廓通过双向桥连接为一条闭合路径。

    RETR_TREE 保留外轮廓、孔、孔内岛及更深的嵌套。
    保持 OpenCV 原始绕行方向。每次将尚未接入的最近轮廓接入已有路径：
        ... A, B, [沿新轮廓走一圈], B, A, ...
    环边界被走一次，桥被正反各走一次。fillPoly 的奇偶填充保留孔内部。
    对碎块也采用同样方式，所以同一个实例始终只产生一行 TXT。
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("空实例")
    rings = [c.reshape(-1, 2).astype(np.int64) for c in contours]
    depth = []
    for i in range(len(rings)):
        d, parent = 0, int(hierarchy[0, i, 3])
        while parent >= 0:
            d += 1
            parent = int(hierarchy[0, parent, 3])
        depth.append(d)
    first = max(range(len(rings)), key=lambda i: abs(cv2.contourArea(contours[i])))
    merged = rings[first].copy()
    remaining = [r for i, r in enumerate(rings) if i != first]
    while remaining:
        best = None
        for k, ring in enumerate(remaining):
            dist, a, b = nearest_pair(merged, ring)
            if best is None or dist < best[0]:
                best = (dist, k, a, b)
        _, k, a, b = best
        ring = remaining.pop(k)
        ring = np.concatenate((ring[b:], ring[:b]))
        merged = np.concatenate((merged[:a + 1], ring, ring[:1], merged[a:a + 1], merged[a + 1:]))
    points = compact_exact(merged)
    if len(np.unique(points, axis=0)) < 3:
        if not allow_subpixel_fallback:
            raise ValueError("亚像素轮廓仍不足三个不同顶点")
        # 极小可见实例可能只有一个像素或一条直线。像素中心轮廓会退化，
        # 但像素本身有面积：在局部 4 倍网格上提取轮廓，再映射回原像素内部。
        # 不删除实例、不扩大成 bbox、不改变其他实例和正常孔洞的编码。
        yy, xx = np.nonzero(binary)
        x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
        crop = binary[y0:y1 + 1, x0:x1 + 1]
        scale = 4
        enlarged = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
        fine, _ = encode_instance(enlarged, allow_subpixel_fallback=False)
        # make_line 还会 +0.5，因此这里减去 0.5，使最终坐标为
        # (细网格坐标 + 0.5)/4 + 裁剪原点，即原像素内部的 .125 至 .875。
        points = (fine.astype(np.float64) + 0.5) / scale + [x0, y0] - 0.5
        fallback_used = 1
    else:
        fallback_used = 0
    return points, {"rings": len(rings), "hole_rings": sum(d % 2 for d in depth),
                    "foreground_rings": sum(d % 2 == 0 for d in depth), "points": len(points),
                    "subpixel_fallback": fallback_used}


def make_line(class_id, points, width, height):
    # +0.5 使用像素中心，防止十进制/float32 往返后
    # 20 变成 19.999998，再经 Ultralytics int32 截断成 19。
    # 像素中心也减轻斜边重采样后截断偏向一侧、侵入小孔的问题。
    # 原尺寸下仍落在相同像素；不声称任意缩放后像素完全一致。
    normalized = (points.astype(np.float64) + 0.5) / [width, height]
    return str(class_id) + " " + " ".join(f"{v:.10f}" for v in normalized.ravel())


def parse_line(line):
    return np.asarray(line.split()[1:], dtype=np.float32).reshape(-1, 2)


def rasterize(normalized, shape):
    height, width = shape
    points = (normalized * np.array([width, height], dtype=np.float32)).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def compare_masks(source, restored):
    src, dst = source.astype(bool), restored.astype(bool)
    union = np.count_nonzero(src | dst)
    # 找不与图像边界相接的背景连通域，逐孔评估；不能仅用全图 IoU 判断小孔。
    n, labels = cv2.connectedComponents((~src).astype(np.uint8), connectivity=4)
    outside = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    sizes = np.bincount(labels.ravel(), minlength=n)
    filled = np.bincount(labels[dst], minlength=n)
    holes = [i for i in range(1, n) if i not in outside]
    recalls = [1.0 - float(filled[i]) / int(sizes[i]) for i in holes]
    return {"iou": float(np.count_nonzero(src & dst) / union) if union else 1.0,
            "added_pixels": int(np.count_nonzero(dst & ~src)),
            "missing_pixels": int(np.count_nonzero(src & ~dst)),
            "holes": len(holes), "hole_pixels_filled": int(sum(filled[i] for i in holes)),
            "holes_completely_filled": sum(r == 0.0 for r in recalls),
            "min_hole_retention": min(recalls, default=1.0)}


def preview(path, source, restored, training=None):
    panels = [("SOURCE", source), ("TXT RESTORED", restored)]
    if training is not None:
        panels.append(("ULTRALYTICS", training))
    rendered = []
    for title, mask in panels:
        tile = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # 控制预览尺寸；最近邻缩放仅影响预览，不参与验证。
        scale = min(4.0, 480 / max(mask.shape))
        tile = cv2.resize(tile, (max(160, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(tile, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(tile)
    save_png(path, np.concatenate(rendered, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True,
                        help="Directory containing seg_instance/ and seg_instance_labels/.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Empty directory used for YOLO labels and validation reports.")
    parser.add_argument("--class-map", default="1:0,2:1,3:2",
                        help="Category mapping, e.g. 1:0,2:1,3:2.")
    parser.add_argument("--verify-ultralytics", action="store_true", help="调用本机 Ultralytics 重采样和 polygon2mask 校验")
    parser.add_argument("--mask-ratio", type=int, default=4, help="额外检查此倍率的 mask 下采样，默认 4")
    parser.add_argument("--preview-limit", type=int, default=30)
    args = parser.parse_args()
    try:
        class_mapping = {int(k): int(v) for k, v in (item.split(":", 1) for item in args.class_map.split(","))}
    except Exception as exc:
        parser.error(f"Invalid --class-map: {exc}")
    if args.mask_ratio < 1 or args.preview_limit < 0:
        parser.error("mask-ratio must be >= 1 and preview-limit must be >= 0")
    mask_dir, json_dir = args.scene_root / "seg_instance", args.scene_root / "seg_instance_labels"
    if not mask_dir.is_dir() or not json_dir.is_dir():
        parser.error("scene-root must contain seg_instance/ and seg_instance_labels/")
    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        parser.error("No instance PNG files were found")
    ultra = None
    if args.verify_ultralytics:
        import ultralytics
        from ultralytics.utils.ops import resample_segments
        from ultralytics.data.utils import polygon2mask
        ultra = (resample_segments, polygon2mask)
        print("Ultralytics version:", ultralytics.__version__)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("output-dir must be empty")
    labels_dir, previews_dir = out / "labels", out / "previews"
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    rows, errors, converted, warnings, shown = [], [], [], [], 0
    duplicate_total = 0
    for index, path in enumerate(masks, 1):
        image_rows = []
        try:
            mask = read_mask(path)
            height, width = mask.shape
            records, duplicates = load_records(json_dir / (path.stem + ".json"))
            duplicate_total += duplicates
            lines, entries = [], []
            for iid, record in records.items():
                cid = int(record["category_id"])
                if cid not in class_mapping:
                    warnings.append(f"{path.name}: instance {iid}, unmapped category {cid}, skipped")
                    continue
                source = (mask == iid).astype(np.uint8)
                visible = int(source.sum())
                if not visible:
                    warnings.append(f"{path.name}: instance {iid} has no visible pixels, skipped")
                    continue
                try:
                    points, info = encode_instance(source)
                except ValueError as exc:
                    raise ValueError(f"instance_id={iid}, visible_pixels={visible}: {exc}") from exc
                line = make_line(class_mapping[cid], points, width, height)
                lines.append(line)
                entries.append((iid, record, source, info, parse_line(line)))
            # 与常见 YOLO11 数据加载器一致：同图实例用相同的重采样点数。
            max_len = max((len(e[4]) for e in entries), default=0)
            resample_count = max_len + 1 if max_len > 1000 else 1000
            for iid, record, source, info, segment in entries:
                restored = rasterize(segment, mask.shape)
                row = {"image": path.name, "instance_id": iid,
                       "visible_pixels": int(source.sum()), **info}
                raw = compare_masks(source, restored)
                row.update({"txt_" + k: v for k, v in raw.items()})
                expected = record.get("visible_pixels_in_instance_segmap")
                if expected is not None and int(expected) > 0:
                    row["visible_pixel_relative_error"] = abs(int(source.sum()) - int(expected)) / int(expected)
                    if row["visible_pixel_relative_error"] > .02:
                        warnings.append(f"{path.name}: instance {iid}, visible pixel count mismatch")
                training = None
                if ultra:
                    sampled = ultra[0]([segment.copy()], n=resample_count)[0]
                    pixels = sampled * np.array([width, height], dtype=np.float32)
                    training = ultra[1](mask.shape, [pixels.reshape(-1)], color=1, downsample_ratio=1)
                    checks = compare_masks(source, training)
                    row.update({"ultra_" + k: v for k, v in checks.items()})
                    if min(mask.shape) < args.mask_ratio:
                        raise ValueError("图像尺寸小于 mask-ratio")
                    reduced = ultra[1](mask.shape, [pixels.reshape(-1)], color=1, downsample_ratio=args.mask_ratio)
                    # 下采样 GT 也用相同的 OpenCV 插值，以分开测量编码误差与分辨率损失。
                    target = cv2.resize(source, (width // args.mask_ratio, height // args.mask_ratio))
                    row.update({"downsample_" + k: v for k, v in compare_masks(target, reduced).items()})
                    row["downsample_source_holes"] = compare_masks(target, target)["holes"]
                    if row["downsample_source_holes"] < raw["holes"]:
                        warnings.append(f"{path.name}: instance {iid}, source holes reduced after mask_ratio={args.mask_ratio} downsampling")
                image_rows.append(row)
                if raw["added_pixels"] or raw["missing_pixels"] or (ultra and (row["ultra_added_pixels"] or row["ultra_missing_pixels"])):
                    warnings.append(f"{path.name}: instance {iid}, reconstruction differs; see report.csv")
                if shown < args.preview_limit:
                    preview(previews_dir / f"{path.stem}_id{iid}.png", source, restored, training)
                    shown += 1
                # 严重错误不能悄悄变成正常训练标签。整张图不输出 TXT。
                if raw["missing_pixels"] or raw["holes_completely_filled"]:
                    raise ValueError(f"instance {iid}: TXT 丢失前景或完全填掉孔洞，需检查预览")
                if ultra and row["ultra_holes_completely_filled"]:
                    raise ValueError(f"instance {iid}: Ultralytics 重采样后孔洞完全消失，需检查版本/读取方式")
            (labels_dir / (path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            converted.append(path.name)
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            errors.append({"image": path.name, "error": str(exc)})
            print(f"[ERROR] {path.name}: {exc}")
        finally:
            for row in image_rows:
                row["image_exported"] = path.name in converted
            rows.extend(image_rows)
        if index % 50 == 0 or index == len(masks):
            print(f"Processed {index}/{len(masks)}")
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["image", "instance_id"]
    with open(out / "report.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    orphan_json = sorted(p.name for p in json_dir.glob("*.json") if p.stem not in {m.stem for m in masks})
    summary = {"images": len(masks), "converted": len(converted), "errors": errors,
               "subpixel_fallback_instances": sum(r.get("subpixel_fallback", 0) for r in rows),
               "duplicate_records": duplicate_total, "orphan_json": orphan_json, "warnings": warnings,
               "ultralytics_version": ultralytics.__version__ if ultra else None,
               "mask_ratio_checked": args.mask_ratio if ultra else None,
               "validation_scope": "original size, no augmentation; optional actual resample and mask downsample",
               "converted_images": converted}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output: {out.resolve()}")
    print(f"Converted: {len(converted)}, errors: {len(errors)}, warnings: {len(warnings)}")
    print("Review report.csv and previews before training. Failed images must not be treated as background.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
