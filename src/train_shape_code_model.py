"""CLI helper to train the shape code model."""

from . import paths
from .shape_code_generator import ShapeCodeGenerator


def main():
    drone_folder = paths.DRONE_DIR
    bird_folder = paths.BIRD_DIR
    output_path = paths.SHAPE_CODE_MODEL_FILE

    print("Training Shape Code Model...")
    print(f"Drone folder: {drone_folder}")
    print(f"Bird folder: {bird_folder}")

    if not drone_folder.exists() or not bird_folder.exists():
        print("Error: Training folders not found!")
        return

    generator = ShapeCodeGenerator()
    try:
        success = generator.train(
            drone_dir=str(drone_folder),
            bird_dir=str(bird_folder),
            output_path=output_path,
        )

        if success:
            print(f"\nSuccess! Model saved to: {output_path}")
            print(f"Drone templates: {len(generator.drone_shape_codes)}")
            print(f"Bird templates: {len(generator.bird_shape_codes)}")
        else:
            print("\nTraining failed: No valid templates found.")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
