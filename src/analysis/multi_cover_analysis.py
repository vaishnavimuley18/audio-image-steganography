import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input/output paths
input_path = Path("results/multi_cover_results.csv")
df = pd.read_csv(input_path)

project_root = Path.cwd()
# If executed from the project directory, use its results folder.
# Otherwise create a local analysis folder beside the uploaded CSV.
results_dir = project_root / "results"
if not results_dir.exists():
    results_dir = input_path.parent / "results"
analysis_dir = results_dir / "multi_cover_analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Status summary
# -----------------------------
status_summary = (
    df.groupby("status")
      .size()
      .reset_index(name="experiments")
)
status_summary["percentage"] = status_summary["experiments"] / len(df) * 100
status_summary.to_csv(analysis_dir / "status_summary.csv", index=False)

# -----------------------------
# 2. Method summary
# -----------------------------
def count_status(x, status):
    return (x == status).sum()

method_summary = (
    df.groupby("method")
      .agg(
          experiments=("status", "size"),
          successful=("status", lambda x: count_status(x, "SUCCESS")),
          capacity_failures=("status", lambda x: count_status(x, "INSUFFICIENT_CAPACITY")),
          integrity_failures=("status", lambda x: count_status(x, "INTEGRITY_FAILED")),
      )
      .reset_index()
)

method_summary["success_rate_percent"] = (
    method_summary["successful"] / method_summary["experiments"] * 100
)

success_df = df[df["status"] == "SUCCESS"]

method_metrics = (
    success_df.groupby("method")
    .agg(
        avg_psnr=("psnr", "mean"),
        avg_ssim=("ssim", "mean"),
        avg_embedding_time_seconds=("embedding_time_seconds", "mean"),
        avg_extraction_time_seconds=("extraction_time_seconds", "mean"),
        avg_capacity_utilization_percent=("capacity_utilization_percent", "mean"),
        avg_changed_pixel_percentage=("changed_pixel_percentage", "mean"),
        avg_changed_channel_percentage=("changed_channel_percentage", "mean"),
        avg_integrity_percent=("integrity_percent", "mean"),
    )
    .reset_index()
)

method_summary = method_summary.merge(method_metrics, on="method", how="left")
method_summary.to_csv(analysis_dir / "method_summary.csv", index=False)

# -----------------------------
# 3. Payload summary
# -----------------------------
payload_summary = (
    df.groupby(["payload_type", "method"])
      .agg(
          experiments=("status", "size"),
          successful=("status", lambda x: count_status(x, "SUCCESS")),
          capacity_failures=("status", lambda x: count_status(x, "INSUFFICIENT_CAPACITY")),
          integrity_failures=("status", lambda x: count_status(x, "INTEGRITY_FAILED")),
      )
      .reset_index()
)
payload_summary["success_rate_percent"] = (
    payload_summary["successful"] / payload_summary["experiments"] * 100
)
payload_summary.to_csv(analysis_dir / "payload_summary.csv", index=False)

# -----------------------------
# 4. Cover summary
# -----------------------------
cover_summary = (
    df.groupby("cover_name")
      .agg(
          experiments=("status", "size"),
          successful=("status", lambda x: count_status(x, "SUCCESS")),
          capacity_failures=("status", lambda x: count_status(x, "INSUFFICIENT_CAPACITY")),
          integrity_failures=("status", lambda x: count_status(x, "INTEGRITY_FAILED")),
      )
      .reset_index()
)
cover_summary["success_rate_percent"] = (
    cover_summary["successful"] / cover_summary["experiments"] * 100
)
cover_summary.to_csv(analysis_dir / "cover_summary.csv", index=False)

# -----------------------------
# 5. Failure analysis
# -----------------------------
failure_analysis = (
    df[df["status"] != "SUCCESS"]
    .groupby(["method", "payload_type", "status"])
    .size()
    .reset_index(name="failure_count")
)
failure_analysis.to_csv(analysis_dir / "failure_analysis.csv", index=False)

# -----------------------------
# 6. Candidate-level analysis
# -----------------------------
# Add normalized helper fields without changing the original CSV.
analyzed = df.copy()
analyzed["eligible"] = (
    (analyzed["status"] == "SUCCESS") &
    (analyzed["capacity_fit"] == True) &
    (analyzed["integrity_percent"] >= 99.999)
)

# Save the enriched experiment table.
analyzed.to_csv(analysis_dir / "analyzed_experiment_results.csv", index=False)

# -----------------------------
# 7. Success matrix
# -----------------------------
success_matrix = (
    df.groupby(["payload_type", "method"])["status"]
      .apply(lambda x: (x == "SUCCESS").sum())
      .unstack(fill_value=0)
)
success_matrix.to_csv(analysis_dir / "payload_method_success_matrix.csv")

# -----------------------------
# 8. Plots
# -----------------------------
def save_bar_plot(x, y, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(analysis_dir / filename, dpi=200)
    plt.close()

save_bar_plot(
    method_summary["method"],
    method_summary["success_rate_percent"],
    "Success Rate by Method",
    "Success Rate (%)",
    "success_rate_by_method.png",
)

save_bar_plot(
    method_metrics["method"],
    method_metrics["avg_psnr"],
    "Average PSNR of Successful Experiments",
    "PSNR (dB)",
    "psnr_by_method.png",
)

save_bar_plot(
    method_metrics["method"],
    method_metrics["avg_ssim"],
    "Average SSIM of Successful Experiments",
    "SSIM",
    "ssim_by_method.png",
)

save_bar_plot(
    method_metrics["method"],
    method_metrics["avg_embedding_time_seconds"],
    "Average Embedding Time",
    "Time (seconds)",
    "embedding_time_by_method.png",
)

save_bar_plot(
    method_metrics["method"],
    method_metrics["avg_extraction_time_seconds"],
    "Average Extraction Time",
    "Time (seconds)",
    "extraction_time_by_method.png",
)

# -----------------------------
# 9. Console summary
# -----------------------------
print("=" * 65)
print("MULTI-COVER ANALYSIS COMPLETE")
print("=" * 65)
print(f"Dataset shape : {df.shape}")
print(f"Output folder : {analysis_dir}")
print()
print(method_summary[
    ["method", "experiments", "successful",
     "capacity_failures", "integrity_failures",
     "success_rate_percent", "avg_psnr", "avg_ssim"]
].round(4).to_string(index=False))

print("\nOverall status:")
print(status_summary.round(2).to_string(index=False))

print("\nGenerated files:")
for f in sorted(analysis_dir.iterdir()):
    print(" -", f.name)
