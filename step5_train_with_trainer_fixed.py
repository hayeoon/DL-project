# step5b_train_manual_fixed.py
# 목적: 교재 파일(Trainer 등) 수정 없이, 동일 네트워크/옵티마이저로
#       간단 학습 루프를 우리 스크립트에서만 돌린다. (TF 사용 안 함)

import os, sys, json, numpy as np
from os.path import dirname, abspath, join

# import 경로 (common/ 혹은 현재 폴더 둘 다 대응)
sys.path.append(join(dirname(abspath(__file__)), "common"))
try:
    from simple_convnet import SimpleConvNet
except Exception:
    from simple_convnet import SimpleConvNet
try:
    from common.optimizer import Adam
except Exception:
    from common.optimizer import Adam

# -------- 하드코딩 경로/파라미터 --------
NPZ_PATH   = "out_step3/dataset.npz"
VOCAB_JSON = "out_step2/vocab.json"
OUT_DIR    = "out_step5_manual"
LR         = 1e-3
ITERS      = 400          # 필요하면 300~800 사이로
LOG_EVERY  = 20
# --------------------------------------

def main():
    if not (os.path.exists(NPZ_PATH) and os.path.exists(VOCAB_JSON)):
        print("[에러] dataset.npz 또는 vocab.json 이 없습니다."); return
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(NPZ_PATH)
    X, y = data["X"], data["y"]             # (N,1,28,28), (N,)
    with open(VOCAB_JSON, "r", encoding="utf-8") as f:
        text2id = json.load(f)
    id2text = {int(v): k for k, v in text2id.items()}
    num_classes = len(text2id)

    N = X.shape[0]
    print(f"[INFO] X:{X.shape} {X.dtype}  y:{y.shape} {y.dtype}  classes:{num_classes}")

    # 네트워크 & 옵티마이저 (교재와 동일 계열)
    net = SimpleConvNet(
        input_dim=(1,28,28),
        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
        hidden_size=100, output_size=num_classes, weight_init_std=0.01
    )
    opt = Adam(lr=LR)

    # 초기 로그
    loss0 = net.loss(X, y)
    pred0 = np.argmax(net.predict(X), axis=1)
    acc0  = (pred0 == y).mean()
    print(f"[BEFORE] loss={loss0:.4f} acc={acc0:.3f}")

    # full-batch 학습 루프 (교재 코드의 gradient/params 그대로 사용)
    for it in range(1, ITERS+1):
        grads = net.gradient(X, y)
        opt.update(net.params, grads)

        if it % LOG_EVERY == 0 or it == 1:
            loss = net.loss(X, y)
            pred = np.argmax(net.predict(X), axis=1)
            acc  = (pred == y).mean()
            print(f"[STEP {it:03d}] loss={loss:.4f} acc={acc:.3f}")

    # 최종 결과 저장
    logits = net.predict(X)
    pred_id = np.argmax(logits, axis=1)
    pred_text = [id2text[int(i)] for i in pred_id]
    true_text = [id2text[int(i)] for i in y]
    match = (pred_id == y)

    # CSV
    csv_path = os.path.join(OUT_DIR, "preds_train.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("idx,true_text,pred_text,match\n")
        for i,(tt,pt,m) in enumerate(zip(true_text, pred_text, match), start=1):
            f.write(f"{i},{tt},{pt},{bool(m)}\n")
    print(f"[SAVE] {csv_path}")
    print(f"[RESULT] manual train acc = {match.mean():.3f}")

    # 모델 파라미터 저장(선택)
    np.savez_compressed(os.path.join(OUT_DIR, "model_params.npz"), **net.params)
    print(f"[SAVE] {os.path.join(OUT_DIR, 'model_params.npz')}")

if __name__ == "__main__":
    main()
