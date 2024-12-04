from ultralytics import YOLO
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Path to the YOLO model (.pt file)
MODEL_PATH = "/Users/rakshitshah/Desktop/Github_projects/RoadShop-Sign-Detection-Android-Application/best.pt"  # Replace with your .pt model path

def load_yolo_model(model_path):
    """
    Load the YOLO model from the given .pt file.
    
    Args:
        model_path (str): Path to the YOLO model (.pt file).
    Returns:
        YOLO: Loaded YOLO model.
    """
    return YOLO(model_path)

def predict_and_visualize(model, image):
    """
    Perform object detection using the YOLO model and visualize the results.
    
    Args:
        model (YOLO): YOLO model instance.
        image (PIL.Image): Input image for detection.
    """
    # Convert the image to RGB if it has an alpha channel (RGBA)
    image_rgb = image.convert("RGB")  # Ensure 3 channels

    # Convert PIL image to numpy array
    image_np = np.array(image_rgb)

    # Run inference
    results = model.predict(source=image_np, save=False, conf=0.5)

    # # Draw bounding boxes and labels
    # draw = ImageDraw.Draw(image_rgb)  # Draw on the RGB image
    # try:
    #     font = ImageFont.truetype("arial.ttf", 20)  # Use a TrueType font (e.g., Arial) with size 20
    # except IOError:
    #     font = ImageFont.load_default()  # Fall

    # for result in results:
    #     for box in result.boxes:
    #         # Extract box coordinates (x1, y1, x2, y2)
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         conf = box.conf[0]  # Confidence score
    #         cls = int(box.cls[0])  # Class ID
    #         label = f"{model.names[cls]} {conf:.2f}"

    #         # Draw bounding box
    #         draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    #         # Add label
    #         draw.text((x1, y1 - 10), label, fill="red")

    # # Display the image with detections
    # st.image(image_rgb, caption="Detected Objects", use_column_width=True)
    # Plot the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')

    for result in results:
        for box in result.boxes:
            # Extract box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            plt.gca().add_patch(
                Rectangle(
                    (x1, y1),  # Bottom-left corner of the rectangle
                    x2 - x1,  # Width
                    y2 - y1,  # Height
                    edgecolor="red",
                    facecolor="none",
                    linewidth=1.5,  # Make the bounding box thicker
                )
            )

            # Add big and bold label on top of the bounding box
            plt.text(
                x1,
                y1 - 15,  # Position label slightly above the bounding box
                label,
                fontsize=12,  # Larger font size
                weight="bold",  # Bold text
                color="white",  # White text color
                bbox=dict(facecolor="red", alpha=0.6, edgecolor="none"),  # Red background
            )
    st.pyplot(plt)


# Streamlit app
def main():
    st.title("YOLO Object Detection")
    st.write("Upload an image, and the YOLO model will perform object detection.")

    # Load the YOLO model
    st.write("Loading YOLO model...")
    model = load_yolo_model(MODEL_PATH)
    st.success("Model loaded successfully!")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform object detection
        st.write("Detecting objects...")
        predict_and_visualize(model, image)

if __name__ == "__main__":
    main()
