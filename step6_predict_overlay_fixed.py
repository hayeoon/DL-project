# step6_predict_overlay_fixed.py
# 목적: 저장한 모델 파라미터로 예측 → (정답, 예측, 일치) CSV + 오버레이 이미지 생성
# TensorFlow 사용 X (NumPy + 교재 simple_convnet)

import os, sys, csv, json, numpy as np
from os.path import dirname, abspath, join
from PIL import Image, ImageDraw, ImageFont

# ====== 경로 하드코딩 ======
IMG_PATH      = "IMG_OCR_53_4PR_09180.png"
LABELS_CSV    = "out_step2/labels_per_bbox.csv"
VOCAB_JSON    = "out_step2/vocab.json"
MODEL_PARAMS  = "out_step5_manual/model_params.npz"
OUT_DIR       = "out_step6"
SIZE, PAD     = 28, 4
# ==========================

# import 경로 (common/ 혹은 현재 폴더 둘 다 대응)
sys.path.append(join(dirname(abspath(__file__)), "common"))
try:
    from simple_convnet import SimpleConvNet
except Exception:
    from simple_convnet import SimpleConvNet

try:
    RESAMPLE = Image.Resampling.BILINEAR
except Exception:
    RESAMPLE = Image.BILINEAR

def load_rows(csv_path):
    rows=[]
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "idx": int(row["idx"]),
                "text_norm": row["text_norm"],
                "class_id": int(row["class_id"]),
                "x1": int(row["x1"]), "y1": int(row["y1"]),
                "x2": int(row["x2"]), "y2": int(row["y2"]),
            })
    rows.sort(key=lambda z: z["idx"])
    return rows

def crop28(img_gray, r, pad=PAD, size=SIZE):
    x1 = max(0, r["x1"] - pad); y1 = max(0, r["y1"] - pad)
    x2 = min(img_gray.width,  r["x2"] + pad); y2 = min(img_gray.height, r["y2"] + pad)
    if x2 <= x1: x2 = min(img_gray.width,  x1 + 1)
    if y2 <= y1: y2 = min(img_gray.height, y1 + 1)
    patch = img_gray.crop((x1, y1, x2, y2)).resize((size, size), RESAMPLE)
    arr = np.asarray(patch, dtype=np.float32) / 255.0  # [0,1]
    return arr[None, :, :]  # (1,28,28)

def main():
    for p in [IMG_PATH, LABELS_CSV, VOCAB_JSON, MODEL_PARAMS]:
        if not os.path.exists(p):
            print("[에러] 파일 없음:", p); return
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 데이터 로드
    rows = load_rows(LABELS_CSV)
    with open(VOCAB_JSON, "r", encoding="utf-8") as f:
        text2id = json.load(f)
    id2text = {int(v): k for k, v in text2id.items()}
    num_classes = len(text2id)

    img = Image.open(IMG_PATH).convert("L")
    X_list, y_list = [], []
    for r in rows:
        X_list.append(crop28(img, r))
        y_list.append(r["class_id"])
    X = np.stack(X_list, axis=0).astype(np.float32)   # (N,1,28,28)
    y = np.array(y_list, dtype=np.int64)

    # 2) 네트워크 만들고 파라미터 로드
    net = SimpleConvNet(
        input_dim=(1,28,28),
        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
        hidden_size=100, output_size=num_classes, weight_init_std=0.01
    )
    params = np.load(MODEL_PARAMS)
    for k in net.params.keys():
        if k in params:
            net.params[k] = params[k]
        else:
            print(f"[경고] {k} 가 모델 파일에 없음(초기값 사용)")

    # 3) 예측
    logits = net.predict(X)
    pred_id = np.argmax(logits, axis=1)
    pred_text = [id2text[int(i)] for i in pred_id]
    true_text = [id2text[int(i)] for i in y]
    match = (pred_id == y)

    # 4) CSV 저장
    csv_path = os.path.join(OUT_DIR, "preds_overlay.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("idx,true_text,pred_text,match\n")
        for i,(tt,pt,m) in enumerate(zip(true_text, pred_text, match), start=1):
            f.write(f"{i},{tt},{pt},{bool(m)}\n")

    # 5) 오버레이 이미지 저장 (초록=정답, 빨강=오답)
    base = Image.open(IMG_PATH).convert("RGB")
    d = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    for r, ok, pt in zip(rows, match, pred_text):
        box = [r["x1"], r["y1"], r["x2"], r["y2"]]
        color = (0, 180, 0) if ok else (220, 0, 0)
        d.rectangle(box, outline=color, width=3)
        d.text((r["x1"]+3, r["y1"]+3), pt, fill=color, font=font)
    out_img = os.path.join(OUT_DIR, "overlay_pred.png")
    base.save(out_img)

    # 6) 요약
    print(f"[RESULT] acc = {match.mean():.3f}  (N={len(rows)})")
    print("[SAVE]", csv_path)
    print("[SAVE]", out_img)

if __name__ == "__main__":
    main()
