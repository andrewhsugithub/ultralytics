from ultralytics import RTDETR


def main():
    print("Loading model architecture and pre-trained weights...")
    model = RTDETR("rtdetr-l-SimAM.yaml").load("rtdetr-l.pt")
    # model = RTDETR("rtdetr-l-HGBlock_SimAM.yaml").load("rtdetr-l.pt")

    print("Starting training...")
    results = model.train(
        data="bccd.yaml",  # Ensure this file is in your directory (from Roboflow)
        epochs=50,  # 50 is a solid baseline for fine-tuning
        # imgsz=640,  # Standard image resolution
        # batch=8,  # Batch size of 8 or 16 is perfect for 12GB VRAM
        device="cuda",
        name="rtdetr_simam_run",  # Sub-folder for this specific experiment
        project="/home/andrew/projects/ultralytics/runs",
        # optimizer="AdamW",  # AdamW is the recommended optimizer for Transformers
        # lr0=0.0001,  # A smaller learning rate is safer for fine-tuning
    )

    print(
        f"Training complete! Best weights saved at: {results.save_dir}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
