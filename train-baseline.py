from ultralytics import RTDETR


def main():
    print("Loading baseline RT-DETR model...")
    model = RTDETR("rtdetr-l.pt")

    print("Starting baseline training...")
    results = model.train(
        data="bccd.yaml",
        epochs=50,
        batch=8,
        device="cuda",
        name="rtdetr_baseline_run",
        project="/home/andrew/projects/ultralytics/runs",
    )

    print(
        f"Baseline training complete! Best weights saved at: {results.save_dir}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
