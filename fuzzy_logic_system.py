import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO

# Load YOLO model
model = YOLO('C:/YOLO_v11/runs/detect/train25/weights/best.pt')

# Define fuzzy variables
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')  # Estimated Distance
brake_light = ctrl.Antecedent([0, 1], 'brake_light')  # Brake light: 0 (OFF), 1 (ON)
target_braking = ctrl.Antecedent(np.arange(0, 101, 1), 'target_braking')  # Target's braking force (from behavior)
your_braking_force = ctrl.Consequent(np.arange(0, 101, 1), 'your_braking_force')  # Your braking force needed

# Membership Functions for distance
distance['close'] = fuzz.trimf(distance.universe, [0, 0, 50])
distance['medium'] = fuzz.trimf(distance.universe, [30, 50, 80])
distance['far'] = fuzz.trimf(distance.universe, [60, 100, 100])

# Membership Functions for braking forces
target_braking['low'] = fuzz.trimf(target_braking.universe, [0, 0, 40])
target_braking['moderate'] = fuzz.trimf(target_braking.universe, [30, 50, 70])
target_braking['high'] = fuzz.trimf(target_braking.universe, [60, 100, 100])

your_braking_force['light'] = fuzz.trimf(your_braking_force.universe, [0, 0, 40])
your_braking_force['moderate'] = fuzz.trimf(your_braking_force.universe, [30, 50, 70])
your_braking_force['hard'] = fuzz.trimf(your_braking_force.universe, [60, 100, 100])

# Brake light membership
brake_light['OFF'] = fuzz.trimf(brake_light.universe, [0, 0, 0])
brake_light['ON'] = fuzz.trimf(brake_light.universe, [1, 1, 1])

# Define Rules
rule1 = ctrl.Rule(brake_light['ON'] & distance['close'], your_braking_force['hard'])
rule2 = ctrl.Rule(brake_light['ON'] & distance['medium'], your_braking_force['moderate'])
rule3 = ctrl.Rule(brake_light['OFF'] & distance['close'], your_braking_force['light'])
rule4 = ctrl.Rule(distance['far'], your_braking_force['light'])
rule5 = ctrl.Rule(target_braking['high'], your_braking_force['hard'])  # Target braking influences

# Control System
braking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
braking_sim = ctrl.ControlSystemSimulation(braking_ctrl)

# YOLO predictions
results = model.predict(source='C:/YOLO_v11/test_image.jpg', conf=0.2)

# Loop through detections
for result in results:
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Extract bounding box as numpy array
        bbox_width = xyxy[2] - xyxy[0]  # Calculate width from x-coordinates
        
        cls = int(box.cls[0])  # Class ID
        estimated_distance = max(0, 100 - bbox_width)  # Simplified estimate
        
        # Hypothetically infer the target's braking force based on size or other available data
        target_force = np.clip((bbox_width / 10) * 10, 0, 100)  # Placeholder
        
        # Pass YOLO output to fuzzy logic
        braking_sim.input['brake_light'] = cls
        braking_sim.input['distance'] = estimated_distance
        braking_sim.input['target_braking'] = target_force
        braking_sim.compute()
        
        print(f"Class: {cls}, Estimated Distance: {estimated_distance}, Target Braking: {target_force}%")
        print(f"Recommended Braking Force: {braking_sim.output['your_braking_force']}%")
