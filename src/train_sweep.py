import os

configs = [
    {"hidden": 64, "seq": 24, "lr": 0.001},
    {"hidden": 128, "seq": 48, "lr": 0.0005},
    {"hidden": 256, "seq": 24, "lr": 0.0003},
]

for c in configs:
    os.system(
        f"python src/train_lstm.py --hidden {c['hidden']} --seq {c['seq']} --lr {c['lr']}"
    )
