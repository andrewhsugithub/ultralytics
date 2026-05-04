from ultralytics import RTDETR


def main():
    print("Loading model architecture and pre-trained weights...")
    model = RTDETR("rtdetr-l-SimAM.yaml").load("rtdetr-l.pt")

    print("Starting training...")
    results = model.train(
        data="bccd.yaml",
        epochs=50,
        batch=8,
        device="cuda",
        name="rtdetr_simam_run",
        project="/home/andrew/projects/ultralytics/runs",
    )

    print(
        f"Training complete! Best weights saved at: {results.save_dir}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
