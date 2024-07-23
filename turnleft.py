import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import sys
import time

# Add YOLOv5 utils directory to the system path
yolov5_path = '/home/aicompetition12/unmanned_sales/yolov5'  # Adjust this path to your YOLOv5 directory
sys.path.append(yolov5_path)
from general import non_max_suppression

# Model file path
#model_path = '/home/aicompetition12/unmanned_sales/runs/train/hyp_stay_and_lrf05model/weights/best.pt'
model_path = '/best.pt'
# Load the model
try:
    # Load the locally trained model weights
    checkpoint = torch.load(model_path)
    model = checkpoint['model']  # Extract the model from the checkpoint
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")


# Image directory path for cam4
#image_dir = '/home/aicompetition12/Datasets/4.testset_sample/cam4/'
image_dir = '/cam1'
# Confidence threshold
confidence_threshold = 0.25

# Store detected object confidences
object_confidences = defaultdict(list)

# Image preprocessing
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])
results_list = []

# Analyze images in the cam4 directory and store object confidences
image_paths = list(Path(image_dir).glob('*.jpg'))  # Load only .jpg files
for image_path in image_paths:
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Convert image to half precision if model weights are in half precision
    if next(model.parameters()).dtype == torch.float16:
        image = image.half()
    image_start = time.time()
    print("image loaded")

    # Get results from the model
    with torch.no_grad():  # Disable gradient computation
        results = model(image)
    image_end = time.time()
    total_time = image_end - image_start
    print("results well out")
    print(f"time for one image : {total_time:.2f}")
    
    # Apply Non-Max Suppression (NMS) to filter the results
    detections = non_max_suppression(results, conf_thres=confidence_threshold, iou_thres=0.45, multi_label=True)

    # Convert results to pandas DataFrame
    if len(detections) > 0 and detections[0] is not None:
        detections = detections[0].cpu().numpy()  # Convert to numpy array
        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
        detections_df = pd.DataFrame(detections, columns=columns)

        # Map class indices to class names
        class_names = model.names  # Assuming model.names contains the class names
        detections_df['name'] = detections_df['class'].map(lambda x: class_names[int(x)])

        # Store detected object names and confidences
        for _, row in detections_df.iterrows():
            results_list.append({
                'image': image_path.name,
                'object_name': row['name'],
                'confidence': row['confidence']
            })

print("dataframe made")

# # Determine presence of objects based on average confidence
# objects_present = {}
# for obj, confidences in object_confidences.items():
#     avg_confidence = sum(confidences) / len(confidences)
#     if avg_confidence > confidence_threshold:
#         objects_present[obj] = avg_confidence

# # Print results
# print("Objects present with average confidence above threshold:")
# for obj, avg_conf in objects_present.items():
#     print(f"{obj}: {avg_conf}")

# Save results to a CSV file
output_csv_path = '/objects_cam1.csv'
# pd.DataFrame(list(objects_present.items()), columns=['object_name', 'average_confidence']).to_csv(output_csv_path, index=False)
# Convert results to DataFrame
results_df = pd.DataFrame(results_list)

# Extract numbers from image names for sorting
results_df['image_number'] = results_df['image'].str.extract(r'(\d+)', expand=False).astype(int)

# Sort by image number
results_df = results_df.sort_values(by='image_number')

# Drop the temporary sorting column
results_df = results_df.drop(columns=['image_number'])
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
