# 🧠 City People Detection & Heatmap Visualization
**Machine Learning & Computer Vision • Developed by Islam Demir**

This project performs **real-time people detection**, **district-based density analysis**, and **heatmap visualization** from multiple surveillance videos using YOLO models.  
Each run automatically generates:

- 📊 District-level statistical analysis  
- 🔥 Normalized heatmap showing relative density  
- 🕒 Time-based city density logs  
- 📈 Hourly density trend graphs  
- 📁 Structured JSON reports

The system is designed for **urban monitoring**, **crowd analysis**, and **AI-assisted city optimization research**.

---

## 🚀 Key Features

### 🔍 **1. YOLO-based People Detection**
- Uses Ultralytics YOLOv8 for human detection  
- GPU acceleration support  
- Adjustable confidence threshold  
- Auto-resizing for faster inference  

---

### 🎥 **2. Multi-District Video Processing**
Each district is assigned:
- A video source  
- A grid coordinate  
- A density score (average people per frame)

The system extracts:
- Total detected people  
- Average people per frame  
- Maximum/minimum density  
- Standard deviation (variation)

---

### 🔥 **3. Heatmap Generation**
A normalized 2×2 or custom-sized heatmap is produced showing relative density.

Example:
