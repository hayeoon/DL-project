# step3_verify_consistency_fixed.py
# 목적: step2의 CSV/vocab과 step3의 dataset.npz가 1:1로 일치하는지 자동 검증
import os, csv, json, numpy as np

# === 경로 하드코딩 ===
LABELS_CSV = "out_step2/labels_per_bbox.csv"
VOCAB_JSON = "out_step2/vocab.json"
NPZ_PATH   = "out_step3/dataset.npz"
# ====================

def load_rows(csv_path):
    rows=[]
    with open(csv_path,"r",encoding="utf-8-sig") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({
                "idx": int(row["idx"]),
                "text_norm": row["text_norm"],
                "class_id": int(row["class_id"]),
            })
    rows.sort(key=lambda z:z["idx"])
    return rows

def main():
    for p in [LABELS_CSV, VOCAB_JSON, NPZ_PATH]:
        if not os.path.exists(p):
            print("[에러] 파일 없음:", p); return

    rows = load_rows(LABELS_CSV)
    with open(VOCAB_JSON,"r",encoding="utf-8") as f:
        t2i = json.load(f)
    id2text = {int(v):k for k,v in t2i.items()}

    data = np.load(NPZ_PATH)
    X, y = data["X"], data["y"]

    ok = True
    if len(rows)!=len(y) or X.shape[0]!=len(rows):
        print("[불일치] 샘플 개수: CSV",len(rows)," NPZ",X.shape[0]); ok=False

    # 1:1 매칭 검사
    mism = []
    for i,(r,cid) in enumerate(zip(rows, y)):
        if id2text.get(int(cid), None) != r["text_norm"]:
            mism.append((i+1, int(cid), id2text.get(int(cid)), r["text_norm"]))
            if len(mism) <= 5:
                print(" [불일치 예]", mism[-1])

    if mism:
        print(f"[결과] 불일치 {len(mism)}개 발견"); ok=False

    print("[확인] X.shape:", X.shape, "| y.shape:", y.shape)
    print("[확인] vocab classes:", len(id2text))
    print("[최종]", "OK (완전 일치)" if ok else "불일치 있음")

if __name__=="__main__":
    main()
