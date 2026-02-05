"""
prepare_nnrtidf_efv.py
----------------------
Carga el dataset NNRTI de HIVDB, filtra EFV y reconstruye secuencias
a partir de columnas P1..Pn (posiciones aminoacídicas). Limpia caracteres
no estándar y genera columnas:
 - Sequence_full   : secuencia limpia completa
 - Sequence_512    : primeros 512 AA
 - Sequence_300    : primeros 300 AA
Las posiciones '-' (consenso) se sustituyen por el aminoácido del consenso
si se pasa un archivo FASTA de referencia; si no, se eliminan.
"""
from pathlib import Path
import argparse
import pandas as pd
import re
from typing import Optional
AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))
def read_fasta_one_seq(path: Path) -> str:
   """Lee un FASTA de una secuencia y devuelve el string concatenado."""
   seq = []
   with open(path, "r") as f:
       for line in f:
           if line.startswith(">"):
               continue
           seq.append(line.strip())
   return "".join(seq).upper()
def reconstruct_sequence_from_Pcols(row: pd.Series, consensus: Optional[str]) -> str:
   """Reconstruye una secuencia proteica desde columnas P1..Pn."""
   pcols = [c for c in row.index if re.fullmatch(r"P\d+", c)]
   pcols.sort(key=lambda c: int(c[1:]))
   seq = []
   for idx, col in enumerate(pcols, start=1):
       val = str(row[col]).strip()
       if not val or val == "nan":
           continue
       if "/" in val:  # mezcla tipo I/V → tomar la primera
           val = val.split("/")[0]
       if val == "-":  # consenso
           if consensus and len(consensus) >= idx:
               seq.append(consensus[idx-1])
           else:
               continue
       elif val in {".", "#", "~", "*"}:
           continue
       else:
           aa = val[0].upper()
           if aa in AA20:
               seq.append(aa)
   return "".join(seq)
def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("--input", required=True, help="Ruta al NNRTI_DataSet.txt")
   ap.add_argument("--out", required=True, help="Ruta CSV de salida")
   ap.add_argument("--consensus-fasta", help="FASTA del consenso de RT (recomendado)")
   ap.add_argument("--drug-col", default="Drug")
   ap.add_argument("--drug-name", default="EFV")
   ap.add_argument("--id-col", default="SeqID")
   args = ap.parse_args()
   df = pd.read_csv(args.input, sep="\t", dtype=str).fillna("")
   # Filtrar EFV
   if args.drug_col in df.columns:
       df = df[df[args.drug_col].str.upper().str.contains(args.drug_name.upper())]
   if df.empty:
       raise SystemExit(f"No se encontraron filas con {args.drug_name}")
   consensus = None
   if args.consensus_fasta:
       consensus = read_fasta_one_seq(Path(args.consensus_fasta))
       print(f"Consenso cargado (longitud = {len(consensus)})")
   out_rows = []
   for _, row in df.iterrows():
       seq_full = reconstruct_sequence_from_Pcols(row, consensus)
       if not seq_full:
           continue
       out_rows.append({
           "SeqID": row.get(args.id_col, ""),
           "Sequence_full": seq_full,
           "Sequence_512": seq_full[:512],
           "Sequence_300": seq_full[:300],
       })
   out_df = pd.DataFrame(out_rows)
   out_df.to_csv(args.out, index=False)
   print(f"Secuencias EFV escritas en {args.out} (n={len(out_df)})")
if __name__ == "__main__":
   main()

   """
   Ejecutar así:
   python src/data/prepare_nnrtidf_efv.py \
 --input data/NNRTI_DataSet.txt \
 --out data/efv_sequences.csv \
 --consensus-fasta data/consensus_RT_B.fasta
   """