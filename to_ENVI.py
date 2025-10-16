import os
import numpy as np
import pandas as pd
from simpledbf import Dbf5
from datetime import datetime
from tqdm import tqdm

# GDAL is used for writing ENVI format files
from osgeo import gdal

# ———————————— Parameter Settings ————————————
input_folder = r"E:\A研子时期_数据\9_13MMT\工程-gis\3-31美国时序\重新定义参考系\田块法表格\00所有表格"
output_envi_dat = r"E:\A研子时期_数据\9_13MMT\工程-gis\3-31美国时序\重新定义参考系\田块法表格\000整合结果.dat"
output_envi_hdr = output_envi_dat.replace('.dat', '.hdr')

# ———————————— Function to extract DOY (Day of Year) from filename ————————————
def get_doy_from_filename(filename):
    try:
        name = os.path.splitext(os.path.basename(filename))[0]
        date_part = name[:5]
        date = datetime.strptime(date_part, '%m-%d')
        return date.timetuple().tm_yday
    except:
        return None

# ———————————— Read all DBF files and build DOY→DataFrame mapping ————————————
doy_to_df = {}
all_dltbs = set()

for fname in os.listdir(input_folder):
    if not fname.lower().endswith('.dbf'):
        continue
    doy = get_doy_from_filename(fname)
    if doy is None:
        print(f"⚠️ Skipped file with unrecognized date: {fname}")
        continue

    try:
        dbf_path = os.path.join(input_folder, fname)
        dbf = Dbf5(dbf_path, codec='utf-8')
        df = dbf.to_dataframe()

        if 'DLTB' not in df.columns or 'MEAN' not in df.columns:
            print(f"⚠️ File missing required fields DLTB/MEAN: {fname}")
            continue

        df = df[['DLTB', 'MEAN']].dropna(subset=['DLTB'])
        df['DLTB'] = df['DLTB'].astype(int)
        all_dltbs.update(df['DLTB'].unique())
        doy_to_df[doy] = df
    except Exception as e:
        print(f"❌ Error reading {fname}: {e}")

# ———————————— Initialize the result DataFrame ————————————
dltb_list = sorted(all_dltbs)
num_samples = len(dltb_list)
num_columns = 1 + 365  # DLTB + 365 DOY columns

# Initialize an empty DataFrame filled with NA values
doy_columns = [f'DOY{doy}' for doy in range(1, 366)]
empty_df = pd.DataFrame(pd.NA, index=range(num_samples), columns=doy_columns)
result_df = pd.DataFrame({'DLTB': dltb_list})
result_df = pd.concat([result_df, empty_df], axis=1)

# ———————————— Fill data into the result DataFrame ————————————
for doy in tqdm(range(1, 366), desc="📅 Building time series table"):
    col_name = f'DOY{doy}'
    if doy in doy_to_df:
        df = doy_to_df[doy][['DLTB', 'MEAN']].copy()
        df['DLTB'] = df['DLTB'].astype(int)
        df = df.rename(columns={'MEAN': col_name})
        result_df = result_df.merge(df, on='DLTB', how='left', suffixes=('', '_new'))
        result_df[col_name] = result_df[f'{col_name}_new'].combine_first(result_df[col_name])
        result_df.drop(columns=[f'{col_name}_new'], inplace=True)

# ———————————— Convert to numpy array, use float32 for consistency ————————————
# DLTB is originally int, but converting to float32 keeps consistent ENVI format
data_array = result_df.to_numpy(dtype=np.float32)  # shape = (num_samples, 366)

# ———————————— Write to ENVI binary (.dat) file ————————————
data_array.tofile(output_envi_dat)

print(f"✅ ENVI data saved: {output_envi_dat}")
print(f"Data shape: {data_array.shape} (rows = samples, columns = DLTB + 365 time steps)")

# ———————————— Generate ENVI header (.hdr) file ————————————
with open(output_envi_hdr, 'w') as hdr:
    hdr.write("ENVI\n")
    hdr.write(f"samples = {num_columns}\n")  # Number of columns (DLTB + 365)
    hdr.write(f"lines   = {num_samples}\n")  # Number of rows (samples)
    hdr.write("bands   = 1\n")               # Number of bands (2D table = 1 band)
    hdr.write("header offset = 0\n")
    hdr.write("file type = ENVI Standard\n")
    hdr.write("data type = 4\n")             # float32
    hdr.write("interleave = bip\n")          # Band interleaved by pixel (2D matrix)
    hdr.write("byte order = 0\n")            # Little endian
    hdr.write("band names = { DLTB, " + ", ".join(doy_columns) + " }\n")

print(f"✅ ENVI header file generated: {output_envi_hdr}")
