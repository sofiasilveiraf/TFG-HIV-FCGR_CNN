import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="CSV con id, png_path, label")

    ap.add_argument("--out", required=True, help="Carpeta de salida")

    ap.add_argument("--val-size", type=float, default=0.15)

    ap.add_argument("--test-size", type=float, default=0.15)

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    print(f"Cargado {len(df)} ejemplos desde {args.input}")

    # 1) split principal train / temp

    train_df, temp_df = train_test_split(

        df, 

        test_size=args.val_size + args.test_size,

        stratify=df["label"],

        random_state=42,

        shuffle=True

    )

    # 2) dividir temp → val / test

    rel_test = args.test_size / (args.val_size + args.test_size)

    val_df, test_df = train_test_split(

        temp_df,

        test_size=rel_test,

        stratify=temp_df["label"],

        random_state=42,

        shuffle=True

    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_csv(f"{args.out}/train.csv", index=False)

    val_df.to_csv(f"{args.out}/val.csv", index=False)

    test_df.to_csv(f"{args.out}/test.csv", index=False)

    print("Guardado en:", args.out)


if __name__ == "__main__":

    main()
 

"""
 Ejecutar así:
 Para 300aa:
 
 mkdir -p data/fcgr_300_split
python src/cgr_fcgr/split_fcgr.py \
 --input data/fcgr/index_300.csv \
 --out data/fcgr_300_split \
 --val-size 0.15 \
 --test-size 0.15

 Para 512 aa:
 mkdir -p data/fcgr_512_split
python src/cgr_fcgr/split_fcgr.py \
 --input data/fcgr/index_512.csv \
 --out data/fcgr_512_split \
 --val-size 0.15 \
 --test-size 0.15
 """