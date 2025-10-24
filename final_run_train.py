import json, unicodedata, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from common.simple_convnet import SimpleConvNet
from common.trainer import Trainer

# ---------------------- 설정 ----------------------
DATA_DIR   = Path("data")      # 입력 경로
SIZE       = 48 
PAD        = 4 #패딩
LR         = 1e-3 #학습률
EPOCHS     = 20 
SEED       = 0

MAX_TRAIN_IMAGES = 300   
MAX_TEST_IMAGES  = 150   

MIN_FREQ = 3     # 이 횟수 미만으로 등장하는 라벨은 버림
TOP_K    = 200     # 빈도 상위 K개만 사용 

try:
    RESAMPLE = Image.Resampling.BILINEAR
except Exception:
    RESAMPLE = Image.BILINEAR

DEBUG = True  

SAVE_COMPRESSED = False     
WRITE_CSV = True            # 필요 없으면 False
CSV_MAX_ROWS = 100_000      # CSV가 너무 크면 상한


def _md_escape(s):
    return str(s).replace("|", "\\|")
def print_topk_label_freq(train_bbox, text_id, k=20):
    """훈련에 실제로 쓰이는 라벨들만 빈도 집계해서 Top-K 표 출력"""
    counts = Counter(r["text_norm"] for r in train_bbox if r["text_norm"] in text_id)
    rows = counts.most_common(k)
    print(f"\n# Top-{k} labels in TRAIN (by frequency)")
    print("| rank | label | count |")
    print("| ---: | :---- | ----: |")
    for i, (t, c) in enumerate(rows, 1):
        print(f"| {i} | {_md_escape(t)} | {c} |")
    return counts  # 필요하면 바깥에서 재활용

def norm_text(s: str) -> str:
    """입력 문자열 s를 유니코드 정규화(NFKC)로 바꾸고, 양끝 공백을 제거해서 돌려준다"""
    return unicodedata.normalize("NFKC", s or "").strip()

def debug_dataset(name, X, y, text_id, topk=10):
    """X/y 모양·범위·결측·클래스 커버리지·상위 클래스 분포 리포트"""
    if X is None or y is None:
        print(f"[DEBUG] {name}: X/y 없음 — 스킵")
        return
    n = len(y)
    print(f"[DEBUG] {name}: X{X.shape} {X.dtype}, y{y.shape} {y.dtype}")

    # 값 범위/NaN 검사
    x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
    has_nan = bool(np.isnan(X).any())
    print(f"[DEBUG] {name}: X range=({x_min:.3f}, {x_max:.3f}), NaN={has_nan}")

    # 라벨 범위/커버리지
    num_classes = len(text_id)
    bad_low  = int(y.min()) < 0
    bad_high = int(y.max()) >= num_classes
    print(f"[DEBUG] {name}: classes={num_classes}, y.min={int(y.min())}, y.max={int(y.max())}, out_of_range={bad_low or bad_high}")

    uniq, counts = np.unique(y, return_counts=True)
    covered = len(uniq)
    cover_pct = 100.0 * covered / max(1, num_classes)
    print(f"[DEBUG] {name}: covered_classes={covered}/{num_classes} ({cover_pct:.1f}%)")

    # Top-k 라벨 분포 (라벨 문자열까지)
    id2text = {i: t for t, i in text_id.items()}
    top_idx = np.argsort(-counts)[:topk]
    top_items = [(int(uniq[i]), int(counts[i]), id2text.get(int(uniq[i]), "?")) for i in top_idx]
    print(f"[DEBUG] {name}: top-{topk} labels:", [f"{lbl}:{cnt}:{txt}" for lbl, cnt, txt in top_items])
def debug_oov(tag, kept_n, unk_n):
    total = kept_n + unk_n
    rate = 100.0 * unk_n / total if total > 0 else 0.0
    print(f"[DEBUG] {tag}: kept={kept_n}, UNK_dropped={unk_n} ({rate:.1f}%)")

def debug_check_orphans(split: str, DATA_DIR: Path):
    root = DATA_DIR / split
    json_stems = {p.stem for p in root.glob("*.json")}
    png_stems  = {p.stem for p in root.glob("*.png")}
    PNG_stems  = {p.stem for p in root.glob("*.PNG")}  # 대문자 확장자까지
    img_stems  = png_stems | PNG_stems

    both   = json_stems & img_stems
    only_j = json_stems - img_stems
    only_i = img_stems  - json_stems
    print(f"[CHECK:{split}] json={len(json_stems)} img={len(img_stems)} both={len(both)} only_json={len(only_j)} only_img={len(only_i)}")
    if only_j:
        print("  - PNG 없음 예:", list(only_j)[:5])
    if only_i:
        print("  - JSON 없음 예:", list(only_i)[:5])

def make_pair(split: str):
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
            print(f"[WARN] PNG 없음: {ppath}") #png가 없으면 샘플 제외
    return pairs

def load_bboxes(json_path: Path):
    """json 에서 텍스트와 바운딩박스 좌표를 꺼내서, 
    모델이 바로 쓰기 위한 형태(정규화된 텍스트, 축정렬 사각형)으로 변환해 list 반환"""
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

def crop(img_gray: Image.Image, r: dict):
    """bbox로 지정된 영역을 size*size 입력 텐서(채널 1)로 표준화"""
    # 패딩 적용 : 여백 포함 크롭
    x1 = max(0, r["x1"] - PAD); y1 = max(0, r["y1"] - PAD)
    x2 = min(img_gray.width,  r["x2"] + PAD); y2 = min(img_gray.height, r["y2"] + PAD)
    
    # 예외 방지 : 최소 1픽셀 보장
    if x2 <= x1: x2 = min(img_gray.width,  x1 + 1)
    if y2 <= y1: y2 = min(img_gray.height, y1 + 1)
    
    patch = img_gray.crop((x1, y1, x2, y2)).resize((SIZE, SIZE), RESAMPLE)
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    return arr[None, :, :] 

def index_labels(train_rows):
    """text_id = ([(텍스트, idx), ...]), text_freqs = ([(텍스트, 빈도), ...])"""
    # train_rows: [{text_norm:..., ...}, ...] 전체 합 = 빈도수 집계
    count = Counter([r["text_norm"] for r in train_rows])
    #정렬 규칙 : 빈도 내림차순, 동일 빈도(사전 오름차순)
    text_freqs = sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))  
    #텍스트 라벨 -> 정수ID 매핑 만들기
    text_id = {t: i for i, (t, _) in enumerate(text_freqs)}
    return text_id, text_freqs
    # text_id: Dict[str, int]
    # text_freqs: List[Tuple[str, int]]

def build_npz(rows, text_id, src_to_img):
    """rows: bbox , [{'src':sample, 'text_norm':..., x1..y2...}], 
    text_id: 정규화 라벨 → 정수 ID 매핑, src_to_img: {'sample': PathToPng}"""
    X_list, y_list = [], []
    unk = 0 #드랍된 샘플 개수

    for r in rows:
        t = r["text_norm"]
        cid = text_id.get(t, None)
        if cid is None: 
            unk += 1
            continue  # test에서 미지 클래스는 제외 
        img_path = src_to_img[r["src"]]     
        img = Image.open(img_path).convert("L")     # 그레이스케일
        X_list.append(crop(img, r))     
        y_list.append(cid)      # 정수 라벨
    
    if not X_list:
        return None, None, unk
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, unk

def main():
    random.seed(SEED); np.random.seed(SEED)

    # 1) 스캔 [(sample, json, png), ...] 
    train_pairs = make_pair("train")  
    test_pairs  = make_pair("test")
    # 파일 매칭 상태 체크
    if DEBUG:
        debug_check_orphans('train', DATA_DIR)
        debug_check_orphans('test',  DATA_DIR)

    # === A안: 이미지 페어 자체 랜덤 샘플링 ===
    if MAX_TRAIN_IMAGES is not None and len(train_pairs) > MAX_TRAIN_IMAGES:
        train_pairs = random.sample(train_pairs, MAX_TRAIN_IMAGES)
    if MAX_TEST_IMAGES is not None and len(test_pairs) > MAX_TEST_IMAGES:
        test_pairs = random.sample(test_pairs, MAX_TEST_IMAGES)

    if not train_pairs:
        print("[ERROR] data/train 에서 (.json, .png) 페어를 찾지 못했습니다."); return
    print(f"[INFO] train samples: {len(train_pairs)} | test samples: {len(test_pairs)}")

    # 2) Train 라벨 모으기
    # train_rows : train 전체 이미지의 모든 bbox 행을 저장하는 리스트
    # src_to_img_train : 원본 PNG 경로 매핑
    train_bbox, train_png = [], {}
    for sample, jpath, ppath in train_pairs: #스캔으로 얻음
        rows = load_bboxes(jpath) #그 이미지의 모든 bbox(라벨/좌표) 리스트를 가져옴
        for i, r in enumerate(rows, start=1):
            #각 bbox 행에 출처 추적용 메타를 붙임
            #src: 어느 이미지에서 온 행인지(예: "IMG_09180").
            #idx: 그 이미지 안에서 몇 번째 bbox인지(1부터).
            r2 = dict(r); r2["src"] = sample; r2["idx"] = i
            train_bbox.append(r2) # 누적 : train 세트의 모든 bbox가 한 리스트에 모임
        train_png[sample] = ppath

    #위에서 모은 train_bbox 전체에서 빈도 집계
    text_id, text_freqs = index_labels(train_bbox)

    if MIN_FREQ is not None or TOP_K is not None:
        kept = [(t,c) for (t,c) in text_freqs if (MIN_FREQ is None or c >= MIN_FREQ)]
        if TOP_K is not None:
            kept = kept[:TOP_K]
        old_size = len(text_id)
        text_id  = {t:i for i,(t,_) in enumerate(kept)}
        print(f"[INFO] filtered vocab: {len(text_id)} kept "
              f"(min_freq≥{MIN_FREQ}, top_k={TOP_K}) | dropped={old_size - len(text_id)} labels")

        # train bbox도 필터된 라벨만 남김 → NPZ에 쓸 표본만 유지
        kept_set = set(text_id.keys())
        train_bbox = [r for r in train_bbox if r["text_norm"] in kept_set]

    print(f"[OK] train rows={len(train_bbox)} | vocab size={len(text_id)}")

    # 3) Train NPZ 생성
    # Xtr: 전처리된 이미지 텐서, ytr: 라벨의 정수 ID 벡터
    Xtr, ytr, unk_tr = build_npz(train_bbox, text_id, train_png)
    if Xtr is None:
        print("[ERROR] train NPZ 생성 실패(모두 UNK)"); return

    # NPZ 저장부
    if SAVE_COMPRESSED:
        np.savez_compressed("train.npz", X=Xtr, y=ytr)
    else:
        np.savez("train.npz", X=Xtr, y=ytr)
    print(f"[SAVE] 'train.npz' | X:{Xtr.shape} y:{ytr.shape} | UNK dropped: {unk_tr}")

    # 4) Test NPZ 생성
    # 한 번만 전처리해서 저장 → 매실행 때 이미지 재크롭/리사이즈를 안 해도 됨
    # 위에서 했던 과정 test 데이터 기준으로 반복 : text_id 만 그대로 사용
    Xte = yte = None
    if test_pairs:
        test_bbox, test_png  = [], {}
        for sample, jpath, ppath in test_pairs:
            rows = load_bboxes(jpath)
            for i, r in enumerate(rows, start=1):
                r2 = dict(r); r2["src"] = sample; r2["idx"] = i
                test_bbox.append(r2)
            test_png [sample] = ppath

        if MIN_FREQ is not None or TOP_K is not None:
            kept_set = set(text_id.keys())
            test_bbox = [r for r in test_bbox if r["text_norm"] in kept_set]

        Xte, yte, unk_te = build_npz(test_bbox, text_id, test_png )
        if Xte is not None:
            # NPZ 저장부
            if SAVE_COMPRESSED:
                np.savez_compressed("test.npz", X=Xte, y=yte)
            else:
                np.savez("test.npz", X=Xte, y=yte)
            print(f"[SAVE] 'test.npz' | X:{Xte.shape} y:{yte.shape} | UNK dropped: {unk_te}")
        else:
            print("[WARN] test NPZ 생성 안 됨(모두 UNK).")

    # 5) 학습 (train에서 만든 text_id 크기에 맞춰 출력 차원 결정)
    num_classes = len(text_id)
    network = SimpleConvNet(
        input_dim=(1, SIZE, SIZE),
        conv_param={'filter_num':16,'filter_size':5,'pad':0,'stride':1},
        hidden_size=256, output_size=num_classes, weight_init_std=0.01
    )
    print(f"[INFO] start training: classes={num_classes}, epochs={EPOCHS}, lr={LR}")
    
    mini_bs = 128 
    has_test = (Xte is not None) and (yte is not None) and (len(yte) > 0)
    x_te = Xte if has_test else Xtr
    t_te = yte if has_test else ytr

    trainer = Trainer(
        network=network, x_train=Xtr, t_train=ytr, x_test=x_te, t_test=t_te,
        epochs=EPOCHS, mini_batch_size=mini_bs,
        optimizer='Adam', optimizer_param={'lr': LR},
        evaluate_sample_num_per_epoch=None   
    )
    trainer.train()

    # 6) 저장
    network.save_params("model_params.pkl")

    # 7) 최종 정확도 요약
    pred_tr = np.argmax(network.predict(Xtr), axis=1)
    acc_tr = float((pred_tr == ytr).mean())

    if Xte is not None and yte is not None and len(yte) > 0:
        pred_te = np.argmax(network.predict(Xte), axis=1)
        acc_te = float((pred_te == yte).mean())
    else:
        acc_te = float("nan")
    print(f"[RESULT] train_acc={acc_tr:.3f} | test_acc={acc_te:.3f}")

    #표 출력
    print_topk_label_freq(train_bbox, text_id, k=20)

if __name__ == "__main__":
    main()
