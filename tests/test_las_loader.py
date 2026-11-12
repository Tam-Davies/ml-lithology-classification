from src.lithology.data.las_loader import LASLoader

las_path = "data/raw/1055867994.las"

loader = LASLoader(las_path)

loader.load()

print("\nWELL METADATA")
print(loader.get_metadata())

print("\nAVAILABLE CURVES")
for curve in loader.get_curves():
    print(curve)

print("\nDATAFRAME PREVIEW")
df = loader.to_dataframe()

print(df.head())
print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())