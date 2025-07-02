# convertnpy.py  ── Conversão PTB-XL WFDB → .npy
# Opção 2: ignora registros "median beats" cujo header tem "/"
import wfdb, numpy as np, pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse

def convert(src_dir: Path, dst_dir: Path, freq_col: str = "filename_hr"):
    """
    Converte todos os .hea/.dat do PTB-XL para .npy (shape: 12×N) float32.
    ─────────────────────────────────────────────────────────────────────
    src_dir : Path
        Diretório raiz que contém ptbxl_database.csv e sub-pastas records100 / records500.
    dst_dir : Path
        Diretório onde os .npy serão salvos.
    freq_col : {"filename_hr", "filename_lr"}
        Coluna do CSV que aponta para o traçado 500 Hz (hr) ou 100 Hz (lr).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    csv_path = src_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path)
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Convertendo"):
        rec_relpath = row[freq_col]          # ex.: records100/00001_lr
        # ── pula median beats (contêm '/')
        if '/' in rec_relpath:
            skipped += 1
            continue

        rec_path = src_dir / rec_relpath
        try:
            rec = wfdb.rdrecord(str(rec_path))
            sig = np.asarray(rec.p_signal, dtype=np.float32).T  # (12, N)
            np.save(dst_dir / f"{row['ecg_id']:05d}.npy", sig)
        except wfdb.HeaderSyntaxError as e:
            print(f"⚠️  pulando {rec_relpath} → {e}")
            skipped += 1

    print(f"✔ Conversão concluída. Arquivos pulados: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Pasta raiz do PTB-XL (contém ptbxl_database.csv).")
    parser.add_argument("--dst", required=True,
                        help="Pasta de destino dos arquivos .npy.")
    parser.add_argument("--freq", choices=["hr", "lr"], default="hr",
                        help="hr = 500 Hz, lr = 100 Hz.")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    freq_col = "filename_hr" if args.freq == "hr" else "filename_lr"

    print(f"→ Convertendo sinais {args.freq.upper()} de\n  {src}\n  para\n  {dst}")
    convert(src, dst, freq_col)
