# step4_forward_sanity_fixed.py
# 목적: dataset.npz 로더 + SimpleConvNet 순전파/역전파 건강검진 (TensorFlow 사용 X)

import os, json, numpy as np
from simple_convnet import SimpleConvNet
from common.optimizer import Adam
#SGD  # 책 코드 스타일: SGD(lr).update(params, grads)

# === 경로 하드코딩 ===
NPZ_PATH   = "out_step3/dataset.npz"
VOCAB_JSON = "out_step2/vocab.json"
BATCH      = 41 #8
ITERS      = 300 #50
LR         = 0.1
# =====================

def main():
    if not os.path.exists(NPZ_PATH) or not os.path.exists(VOCAB_JSON):
        print("[에러] dataset.npz 또는 vocab.json 이 없습니다."); return

    data = np.load(NPZ_PATH)
    X, y = data["X"], data["y"]             # X:(N,1,28,28) float32, y:(N,) int64
    with open(VOCAB_JSON, "r", encoding="utf-8") as f:
        text2id = json.load(f)
    num_classes = len(text2id)

    print(f"[INFO] X:{X.shape} {X.dtype}  y:{y.shape} {y.dtype}  classes:{num_classes}")

    # 작은 배치 선택
    N = X.shape[0]
    idx = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(idx)
    idx = idx[:min(BATCH, N)]
    xb, yb = X[idx], y[idx]

    # 네트워크 구성(네 책 코드 규격)
    net = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100,
        output_size=num_classes,
        weight_init_std=0.01
    )

    # 순전파 손실/정확도
    loss0 = net.loss(xb, yb)
    #acc0  = net.accuracy(xb, yb)
    acc0  = net.accuracy(xb, yb, batch_size=xb.shape[0])
    print(f"[BEFORE] loss={loss0:.4f}  acc={acc0:.3f}")

    # 간단 SGD로 몇 step 업데이트(미니 오버핏 직전 건강검진)
    #opt = SGD(lr=LR)
    opt = Adam(lr=1e-3)
    for it in range(1, ITERS+1):
        grads = net.gradient(xb, yb)        # 역전파로 기울기 계산
        opt.update(net.params, grads)       # 가중치 갱신
        if it % 10 == 0 or it == 1:
            cur_loss = net.loss(xb, yb)
            #cur_acc  = net.accuracy(xb, yb)
            cur_acc  = net.accuracy(xb, yb, batch_size=xb.shape[0])
            print(f"[STEP {it:02d}] loss={cur_loss:.4f}  acc={cur_acc:.3f}")

    # 최종 확인
    loss1 = net.loss(xb, yb)
    acc1  = net.accuracy(xb, yb)
    print(f"[AFTER ] loss={loss1:.4f}  acc={acc1:.3f}")
    print("→ 손실이 내려가고 정확도가 오르면 입력/라벨/역전파 모두 정상입니다.")

    pred = np.argmax(net.predict(xb), axis=1)
    manual_acc = (pred == yb).mean()
    print("manual acc:", manual_acc)

if __name__ == "__main__":
    main()
