# run_train_test_pipeline.py
# 목적: /data/train, /data/test를 자동 스캔하여
#  1) 전역 vocab(Train에서만) 고정
#  2) Train/Test NPZ 생성(28x28)
#  3) 수동 학습 루프(Adam, full-batch)로 모델 학습
#  4) Train/Test 정확도 출력
#
# 요구: Pillow, numpy, (교재) simple_convnet.py, optimizer.py
# 주의: TensorFlow 사용 안 함

import os, json, csv, unicodedata, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# --- 교재 네트워크/옵티마이저 임포트 (common 폴더/현재 폴더 모두 대응) ---
sys.path.append(str(Path(__file__).parent / "common"))
try:
    from simple_convnet import SimpleConvNet
except Exception:
    from simple_convnet import SimpleConvNet
try:
    from common.optimizer import Adam
except Exception:
    from common.optimizer import Adam
try:
    from common.trainer import Trainer
except Exception:
    from common.trainer import Trainer

# ---------------------- 설정 ----------------------
DATA_DIR   = Path("data")      # 구조: data/train/*.json|.png, data/test/*.json|.png
OUT_ROOT   = Path("out_run")   # 산출물 루트
SIZE       = 28
PAD        = 4
LR         = 1e-3
EPOCHS     = 20
SEED       = 0
# -------------------------------------------------

try:
    RESAMPLE = Image.Resampling.BILINEAR
except Exception:
    RESAMPLE = Image.BILINEAR

# ------------------- 유틸/전처리 -------------------
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def scan_split(split: str):
    """data/<split>에서 (sample, json, png) 목록 수집"""
    root = DATA_DIR / split
    if not root.exists():
        return []
    pairs = []
    for jpath in sorted(root.glob("*.json")):
        base = jpath.stem
        ppath = root / f"{base}.png"
        if ppath.exists():
            pairs.append((base, jpath, ppath))
        else:
            print(f"[WARN] PNG 없음: {ppath}")
    return pairs

def load_bboxes(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    # 기대 형식: {"bbox":[{"x":[...],"y":[...],"data":"..."}, ...]}
    out = []
    for bb in j.get("bbox", []):
        xs, ys = bb["x"], bb["y"]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        out.append({
            "text_raw": bb.get("data", ""),
            "text_norm": norm_text(bb.get("data", "")),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return out

def crop28(img_gray: Image.Image, r: dict):
    x1 = max(0, r["x1"] - PAD); y1 = max(0, r["y1"] - PAD)
    x2 = min(img_gray.width,  r["x2"] + PAD); y2 = min(img_gray.height, r["y2"] + PAD)
    if x2 <= x1: x2 = min(img_gray.width,  x1 + 1)
    if y2 <= y1: y2 = min(img_gray.height, y1 + 1)
    patch = img_gray.crop((x1, y1, x2, y2)).resize((SIZE, SIZE), RESAMPLE)
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    return arr[None, :, :]  # (1,28,28)

def make_vocab_from_train(train_rows):
    # train_rows: [{text_norm:..., ...}, ...] 전체 합
    cnt = Counter([r["text_norm"] for r in train_rows])
    vocab_items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))  # 빈도↓, 사전순↑
    text2id = {t: i for i, (t, _) in enumerate(vocab_items)}
    return text2id, vocab_items

def build_npz(rows, text2id, src_to_img):
    """rows: [{'src':sample, 'text_norm':..., x1..y2...}], text2id: train vocab
       src_to_img: {'sample': PathToPng}"""
    X_list, y_list = [], []
    unk = 0
    for r in rows:
        t = r["text_norm"]
        cid = text2id.get(t, None)
        if cid is None:
            unk += 1
            continue  # test에서 미지 클래스는 제외
        img_path = src_to_img[r["src"]]
        img = Image.open(img_path).convert("L")
        X_list.append(crop28(img, r))
        y_list.append(cid)
    if not X_list:
        return None, None, unk
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, unk

'''
def train_fullbatch(net, Xtr, ytr, Xte=None, yte=None, epochs=EPOCHS, lr=LR):
    opt = Adam(lr=lr)
    for ep in range(1, epochs+1):
        grads = net.gradient(Xtr, ytr)
        opt.update(net.params, grads)
        # 로그
        pred_tr = np.argmax(net.predict(Xtr), axis=1)
        tr_acc  = float((pred_tr == ytr).mean())
        tr_loss = float(net.loss(Xtr, ytr))
        if Xte is not None and yte is not None and len(yte) > 0:
            pred_te = np.argmax(net.predict(Xte), axis=1)
            te_acc  = float((pred_te == yte).mean())
        else:
            te_acc = float("nan")
        print(f"epoch {ep:02d} | loss={tr_loss:.3f} | train_acc={tr_acc:.3f} | test_acc={te_acc:.3f}")
'''

# ------------------------ 메인 ------------------------
def main():
    random.seed(SEED); np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(exist_ok=True, parents=True)
    (OUT_ROOT / "test").mkdir(exist_ok=True, parents=True)
    (OUT_ROOT / "npz").mkdir(exist_ok=True, parents=True)
    (OUT_ROOT / "model").mkdir(exist_ok=True, parents=True)

    # 1) 스캔
    train_pairs = scan_split("train")  # [(sample, json, png), ...]
    test_pairs  = scan_split("test")
    if not train_pairs:
        print("[ERROR] data/train 에서 (.json, .png) 페어를 찾지 못했습니다."); return
    print(f"[INFO] train samples: {len(train_pairs)} | test samples: {len(test_pairs)}")

    # 2) Train 라벨 모으기 → 전역 vocab
    train_rows, src_to_img_train = [], {}
    for sample, jpath, ppath in train_pairs:
        rows = load_bboxes(jpath)
        for i, r in enumerate(rows, start=1):
            r2 = dict(r); r2["src"] = sample; r2["idx"] = i
            train_rows.append(r2)
        src_to_img_train[sample] = ppath
    text2id, vocab_items = make_vocab_from_train(train_rows)
    with (OUT_ROOT/"train"/"vocab_train.json").open("w", encoding="utf-8") as f:
        json.dump(text2id, f, ensure_ascii=False, indent=2)
    # (선택) 병합 CSV 저장
    with (OUT_ROOT/"train"/"labels_merged_train.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["src","idx","text_norm","class_id","x1","y1","x2","y2"])
        for r in train_rows:
            w.writerow([r["src"], r["idx"], r["text_norm"], text2id[r["text_norm"]],
                        r["x1"], r["y1"], r["x2"], r["y2"]])
    print(f"[OK] train rows={len(train_rows)} | vocab size={len(text2id)}")
    print("[Top-5]", [f"{t} x{c}" for t, c in vocab_items[:5]])

    # 3) Train NPZ 생성
    Xtr, ytr, unk_tr = build_npz(train_rows, text2id, src_to_img_train)
    if Xtr is None:
        print("[ERROR] train NPZ 생성 실패(모두 UNK)"); return
    np.savez_compressed(OUT_ROOT/"npz"/"train.npz", X=Xtr, y=ytr)
    print(f"[SAVE] {OUT_ROOT/'npz'/'train.npz'} | X:{Xtr.shape} y:{ytr.shape} | UNK dropped: {unk_tr}")

    # 4) Test NPZ 생성 (train vocab으로 매핑)
    Xte = yte = None
    if test_pairs:
        test_rows, src_to_img_test = [], {}
        for sample, jpath, ppath in test_pairs:
            rows = load_bboxes(jpath)
            for i, r in enumerate(rows, start=1):
                r2 = dict(r); r2["src"] = sample; r2["idx"] = i
                test_rows.append(r2)
            src_to_img_test[sample] = ppath
        Xte, yte, unk_te = build_npz(test_rows, text2id, src_to_img_test)
        if Xte is not None:
            np.savez_compressed(OUT_ROOT/"npz"/"test.npz", X=Xte, y=yte)
            print(f"[SAVE] {OUT_ROOT/'npz'/'test.npz'} | X:{Xte.shape} y:{yte.shape} | UNK dropped: {unk_te}")
        else:
            print("[WARN] test NPZ 생성 안 됨(모두 UNK).")

    # 5) 학습 (train vocab 크기에 맞춰 출력 차원 세팅)
    num_classes = len(text2id)
    network = SimpleConvNet(
        input_dim=(1, SIZE, SIZE),
        conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
        hidden_size=100, output_size=num_classes, weight_init_std=0.01
    )
    #print(f"[INFO] start training: classes={num_classes}, epochs={EPOCHS}, lr={LR}")
    # train_fullbatch(net, Xtr, ytr, Xte, yte, epochs=EPOCHS, lr=LR)
    
    print(f"[INFO] start training with Trainer: classes={num_classes}, epochs={EPOCHS}, lr={LR}")
    mini_bs = 64
    has_test = (Xte is not None) and (yte is not None) and (len(yte) > 0)
    x_te = Xte if has_test else Xtr
    t_te = yte if has_test else ytr

    trainer = Trainer(
        network=network,x_train=Xtr, t_train=ytr,x_test=x_te,  t_test=t_te,
        epochs=EPOCHS,mini_batch_size=mini_bs,
        optimizer='Adam', optimizer_param={'lr': LR},
        evaluate_sample_num_per_epoch=min(len(Xtr), 1000)  # 로그가 확실히 찍히도록 
    )
    trainer.train()

    # 6) 저장 (교재 방식 pkl + vocab 스냅샷)
    network.save_params(str(OUT_ROOT/"model"/"model_params.pkl"))
    with (OUT_ROOT/"model"/"vocab.used.json").open("w", encoding="utf-8") as f:
        json.dump(text2id, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {OUT_ROOT/'model'/'model_params.pkl'}")
    print(f"[SAVE] {OUT_ROOT/'model'/'vocab.used.json'}")

    # 7) 최종 정확도 요약
    pred_tr = np.argmax(network.predict(Xtr), axis=1); acc_tr = float((pred_tr == ytr).mean())
    if Xte is not None and yte is not None and len(yte) > 0:
        pred_te = np.argmax(network.predict(Xte), axis=1); acc_te = float((pred_te == yte).mean())
    else:
        acc_te = float("nan")
    print(f"[RESULT] train_acc={acc_tr:.3f} | test_acc={acc_te:.3f}")

if __name__ == "__main__":
    main()
