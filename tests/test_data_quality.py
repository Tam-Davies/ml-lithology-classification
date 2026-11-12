
from pathlib import Path

from src.lithology.data.las_loader import LASLoader
from src.lithology.preprocessing.quality import DataQualityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAS_PATH = PROJECT_ROOT / "data" / "raw" / "1055867994.las"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_quality"


def main():
    loader = LASLoader(LAS_PATH)
    loader.load()

    df = loader.to_dataframe()

    analyzer = DataQualityAnalyzer(
        dataframe=df,
        depth_column="DEPTH",
        null_values=[-999.25],
    )

    report = analyzer.generate_report()

    print("\nDATASET SHAPE")
    print(report["dataset_shape"])

    print("\nMISSING VALUES")
    print(report["missing_values"])

    print("\nDUPLICATE RECORDS")
    print(report["duplicate_records"])

    print("\nDUPLICATE DEPTHS")
    print(report["duplicate_depths"].head(10))

    print("\nINVALID VALUES")
    print(report["invalid_values"])

    print("\nPOTENTIAL OUTLIERS")
    print(report["outliers"])

    print("\nCONSTANT CURVES")
    print(report["constant_curves"])

    print("\nDEPTH CONTINUITY")
    print(report["depth_continuity"])

    print("\nSTATISTICAL SUMMARY")
    print(report["statistics"])

    analyzer.export_report(OUTPUT_PATH)

    print(f"\nReports exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
