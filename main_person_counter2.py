"""
City People Density Analyzer
============================

Advanced multi-district people detection, density analysis and visualization system
using YOLO and OpenCV with professional-grade features.

Author: Islam Demir (Machine Learning & CV Researcher)
Version: 2.0.0
License: MIT
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION DATACLASSES
# =====================================================

@dataclass
class ModelConfig:
    """YOLO model configuration."""
    path: str = "yolov8n.pt"
    confidence_threshold: float = 0.3
    use_gpu: bool = True
    input_size: int = 640
    
    def __post_init__(self):
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "confidence_threshold": self.confidence_threshold,
            "use_gpu": self.use_gpu,
            "input_size": self.input_size
        }

@dataclass
class ProcessingConfig:
    """Video processing configuration."""
    mode: str = "balanced"  # fast, balanced, accurate
    frame_skip: int = 3
    max_frames: Optional[int] = None
    resize_width: int = 640
    batch_size: int = 1
    show_preview: bool = False
    
    def __post_init__(self):
        mode_settings = {
            "fast": {"frame_skip": 5, "resize_width": 416},
            "balanced": {"frame_skip": 3, "resize_width": 640},
            "accurate": {"frame_skip": 1, "resize_width": 1280}
        }
        
        if self.mode in mode_settings:
            settings = mode_settings[self.mode]
            self.frame_skip = settings["frame_skip"]
            self.resize_width = settings["resize_width"]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode,
            "frame_skip": self.frame_skip,
            "max_frames": self.max_frames,
            "resize_width": self.resize_width,
            "batch_size": self.batch_size,
            "show_preview": self.show_preview
        }

@dataclass
class DistrictConfig:
    """Single district configuration."""
    name: str
    video_path: str
    position: Tuple[int, int]
    weight: float = 1.0
    
    def __post_init__(self):
        if not Path(self.video_path).exists():
            logging.warning(f"Video file not found: {self.video_path}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "video_path": self.video_path,
            "position": list(self.position),
            "weight": self.weight
        }

@dataclass
class OutputConfig:
    """Output and visualization configuration."""
    results_dir: str = "results"
    save_results: bool = True
    save_plots: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    dpi: int = 300
    plot_style: str = "default"
    
    def __post_init__(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "results_dir": self.results_dir,
            "save_results": self.save_results,
            "save_plots": self.save_plots,
            "export_formats": self.export_formats,
            "dpi": self.dpi,
            "plot_style": self.plot_style
        }

@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    districts: List[DistrictConfig] = field(default_factory=list)
    grid_shape: Tuple[int, int] = (2, 2)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            processing=ProcessingConfig(**data.get('processing', {})),
            output=OutputConfig(**data.get('output', {})),
            districts=[DistrictConfig(**d) for d in data.get('districts', [])],
            grid_shape=tuple(data.get('grid_shape', [2, 2]))
        )
    
    @classmethod
    def from_json(cls, path: str) -> 'AppConfig':
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            processing=ProcessingConfig(**data.get('processing', {})),
            output=OutputConfig(**data.get('output', {})),
            districts=[DistrictConfig(**d) for d in data.get('districts', [])],
            grid_shape=tuple(data.get('grid_shape', [2, 2]))
        )
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            'model': self.model.to_dict(),
            'processing': self.processing.to_dict(),
            'output': self.output.to_dict(),
            'districts': [d.to_dict() for d in self.districts],
            'grid_shape': list(self.grid_shape)
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

# =====================================================
# LOGGING SETUP
# =====================================================

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Setup enhanced logging with file and console handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"analysis_{timestamp}.log"
    
    logger = logging.getLogger("CityDensityAnalyzer")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    return logger

# =====================================================
# DETECTION RESULTS
# =====================================================

@dataclass
class DistrictStats:
    """Statistics for a single district."""
    name: str
    position: Tuple[int, int]
    total_people: int
    avg_people: float
    max_people: int
    min_people: int
    std_people: float
    median_people: float
    processed_frames: int
    total_frames: int
    fps: float
    duration_seconds: float
    frame_counts: List[int]
    avg_confidence: float
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "name": self.name,
            "position": list(self.position),
            "total_people": self.total_people,
            "avg_people": self.avg_people,
            "max_people": self.max_people,
            "min_people": self.min_people,
            "std_people": self.std_people,
            "median_people": self.median_people,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "avg_confidence": self.avg_confidence,
            "processing_time": self.processing_time,
            "density_category": self.get_density_category()
        }
    
    def get_density_category(self) -> str:
        """Categorize density level."""
        if self.avg_people < 5:
            return "Low"
        elif self.avg_people < 15:
            return "Medium"
        elif self.avg_people < 30:
            return "High"
        else:
            return "Very High"

# =====================================================
# PEOPLE DETECTOR
# =====================================================

class PeopleDetector:
    """YOLO-based person detection with enhanced features."""
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        """Initialize detector with configuration."""
        self.config = config
        self.logger = logger
        self.model: Optional[YOLO] = None
        self.device = "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model with error handling."""
        try:
            self.logger.info(f"Loading YOLO model: {self.config.path}")
            self.model = YOLO(self.config.path)
            
            if self.config.use_gpu:
                try:
                    self.model.to("cuda")
                    self.device = "cuda"
                    self.logger.info("✅ GPU (CUDA) acceleration enabled")
                except Exception as e:
                    self.logger.warning(f"GPU not available: {e}. Using CPU")
                    self.device = "cpu"
            else:
                self.logger.info("ℹ️  Using CPU for inference")
                
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detect people in a single frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Tuple of (person_count, detections_list)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            results = self.model(
                frame,
                verbose=False,
                device=self.device,
                conf=self.config.confidence_threshold
            )
            
            person_count = 0
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    if cls_id == 0:  # COCO class 0 = person
                        person_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            "area": int((x2 - x1) * (y2 - y1))
                        })
            
            return person_count, detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return 0, []

# =====================================================
# VIDEO PROCESSOR
# =====================================================

class VideoProcessor:
    """Enhanced video processing with rich progress tracking."""
    
    def __init__(
        self,
        detector: PeopleDetector,
        config: ProcessingConfig,
        logger: logging.Logger
    ):
        self.detector = detector
        self.config = config
        self.logger = logger
    
    def process_video(
        self,
        video_path: str,
        district_name: str
    ) -> Optional[DistrictStats]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to video file
            district_name: Name of the district
            
        Returns:
            DistrictStats object or None if failed
        """
        if not Path(video_path).exists():
            self.logger.error(f"Video not found: {video_path}")
            return None
        
        start_time = datetime.now()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        duration = total_frames / fps if fps > 0 else 0
        
        process_limit = min(
            total_frames,
            self.config.max_frames or total_frames
        )
        
        self.logger.info(
            f"Processing {district_name}: "
            f"{total_frames} frames @ {fps:.1f} FPS "
            f"({duration:.1f}s)"
        )
        
        frame_counts = []
        all_confidences = []
        frame_idx = 0
        
        with tqdm(
            total=process_limit,
            desc=f"🎥 {district_name}",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="blue"
        ) as pbar:
            
            while frame_idx < process_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on configuration
                if frame_idx % self.config.frame_skip != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Resize frame
                if self.config.resize_width:
                    h, w = frame.shape[:2]
                    scale = self.config.resize_width / w
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (self.config.resize_width, new_h))
                
                # Detect people
                count, detections = self.detector.detect(frame)
                frame_counts.append(count)
                
                if detections:
                    confidences = [d["confidence"] for d in detections]
                    all_confidences.extend(confidences)
                
                # Optional preview
                if self.config.show_preview:
                    self._show_preview(frame, count, detections, district_name)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        if self.config.show_preview:
            cv2.destroyAllWindows()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        if not frame_counts:
            self.logger.warning(f"No frames processed for {district_name}")
            return None
        
        counts_array = np.array(frame_counts)
        
        stats = DistrictStats(
            name=district_name,
            position=(0, 0),  # Will be set by caller
            total_people=int(np.sum(counts_array)),
            avg_people=float(np.mean(counts_array)),
            max_people=int(np.max(counts_array)),
            min_people=int(np.min(counts_array)),
            std_people=float(np.std(counts_array)),
            median_people=float(np.median(counts_array)),
            processed_frames=len(frame_counts),
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            frame_counts=frame_counts,
            avg_confidence=float(np.mean(all_confidences)) if all_confidences else 0.0,
            processing_time=processing_time
        )
        
        self.logger.info(
            f"✅ {district_name}: {stats.total_people:,} people detected "
            f"(avg: {stats.avg_people:.1f}/frame) in {processing_time:.1f}s"
        )
        
        return stats
    
    def _show_preview(
        self,
        frame: np.ndarray,
        count: int,
        detections: List[Dict],
        district_name: str
    ):
        """Display live preview with detections."""
        display = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        cv2.putText(
            display,
            f"{district_name} | People: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        cv2.imshow("Preview (press 'q' to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.config.show_preview = False

# =====================================================
# CITY ANALYZER
# =====================================================

class CityAnalyzer:
    """Main city-level analysis orchestrator."""
    
    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.console = Console()
        
        self.detector = PeopleDetector(config.model, logger)
        self.processor = VideoProcessor(self.detector, config.processing, logger)
    
    def analyze_city(self) -> Tuple[np.ndarray, List[DistrictStats]]:
        """
        Analyze all districts and generate city heatmap.
        
        Returns:
            Tuple of (heatmap_array, district_stats_list)
        """
        self.console.print(Panel.fit(
            "[bold cyan]🏙️  CITY PEOPLE DENSITY ANALYSIS[/bold cyan]",
            border_style="cyan"
        ))
        
        heatmap = np.zeros(self.config.grid_shape, dtype=float)
        all_stats = []
        
        for district_cfg in self.config.districts:
            self.logger.info(f"Processing district: {district_cfg.name}")
            
            stats = self.processor.process_video(
                district_cfg.video_path,
                district_cfg.name
            )
            
            if stats:
                stats.position = district_cfg.position
                all_stats.append(stats)
                
                row, col = district_cfg.position
                heatmap[row, col] = stats.avg_people
        
        self.console.print(
            f"\n[bold green]✅ Analysis completed for {len(all_stats)} districts[/bold green]\n"
        )
        
        return heatmap, all_stats
    
    def save_results(
        self,
        heatmap: np.ndarray,
        stats: List[DistrictStats]
    ) -> Dict[str, str]:
        """
        Save analysis results in multiple formats.
        
        Returns:
            Dictionary of output file paths
        """
        if not self.config.output.save_results:
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        output_files = {}
        
        # JSON Report
        if "json" in self.config.output.export_formats or \
           "all" in self.config.output.export_formats:
            json_path = self._save_json_report(heatmap, stats, timestamp, date_str)
            output_files['json'] = json_path
        
        # CSV Export
        if "csv" in self.config.output.export_formats or \
           "all" in self.config.output.export_formats:
            csv_path = self._save_csv_report(stats, timestamp)
            output_files['csv'] = csv_path
        
        # Excel Export
        if "excel" in self.config.output.export_formats or \
           "all" in self.config.output.export_formats:
            excel_path = self._save_excel_report(stats, timestamp)
            output_files['excel'] = excel_path
        
        return output_files
    
    def _save_json_report(
        self,
        heatmap: np.ndarray,
        stats: List[DistrictStats],
        timestamp: str,
        date_str: str
    ) -> str:
        """Save detailed JSON report."""
        json_path = Path(self.config.output.results_dir) / f"analysis_{timestamp}.json"
        
        report = {
            "metadata": {
                "timestamp": timestamp,
                "date": date_str,
                "version": "2.0.0",
                "grid_shape": list(self.config.grid_shape)
            },
            "configuration": {
                "model": self.config.model.to_dict(),
                "processing": self.config.processing.to_dict()
            },
            "city_summary": {
                "total_districts": len(stats),
                "total_people": int(sum(s.total_people for s in stats)),
                "avg_density": float(np.mean([s.avg_people for s in stats])) if stats else 0.0,
                "max_density": float(max((s.avg_people for s in stats), default=0.0))
            },
            "districts": [s.to_dict() for s in stats],
            "heatmap": heatmap.tolist()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 JSON report saved: {json_path}")
        return str(json_path)
    
    def _save_csv_report(self, stats: List[DistrictStats], timestamp: str) -> str:
        """Save district statistics as CSV."""
        csv_path = Path(self.config.output.results_dir) / f"statistics_{timestamp}.csv"
        
        df = pd.DataFrame([s.to_dict() for s in stats])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.logger.info(f"💾 CSV report saved: {csv_path}")
        return str(csv_path)
    
    def _save_excel_report(self, stats: List[DistrictStats], timestamp: str) -> str:
        """Save comprehensive Excel report with multiple sheets."""
        excel_path = Path(self.config.output.results_dir) / f"report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Districts', 'Total People Detected', 'Average Density', 'Max Density'],
                'Value': [
                    len(stats),
                    sum(s.total_people for s in stats),
                    f"{np.mean([s.avg_people for s in stats]):.2f}" if stats else "0.00",
                    f"{max((s.avg_people for s in stats), default=0.0):.2f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # District details
            df = pd.DataFrame([s.to_dict() for s in stats])
            df.to_excel(writer, sheet_name='Districts', index=False)
        
        self.logger.info(f"💾 Excel report saved: {excel_path}")
        return str(excel_path)

# =====================================================
# VISUALIZATION ENGINE
# =====================================================

class Visualizer:
    """Advanced visualization system with modern aesthetics."""
    
    def __init__(self, config: OutputConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.console = Console()
        
        # Set plot style
        try:
            if config.plot_style != "default":
                plt.style.use(config.plot_style)
        except:
            sns.set_style("darkgrid")
    
    def plot_heatmap(
        self,
        heatmap: np.ndarray,
        districts: List[DistrictConfig],
        timestamp: str = None
    ):
        """Create and save city density heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize heatmap
        max_val = np.max(heatmap)
        norm_heatmap = heatmap / max_val if max_val > 0 else heatmap
        
        # Create heatmap
        sns.heatmap(
            norm_heatmap,
            annot=False,
            cmap="RdYlGn_r",
            cbar_kws={'label': 'Normalized Density', 'shrink': 0.8},
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=2,
            linecolor='white',
            square=True
        )
        
        # Add district labels
        for district in districts:
            name = district.name
            row, col = district.position
            value = heatmap[row, col]
            
            text_color = 'white' if norm_heatmap[row, col] > 0.5 else 'black'
            
            ax.text(
                col + 0.5, row + 0.5,
                f"{name}\n{value:.1f} people/frame",
                ha='center', va='center',
                color=text_color,
                fontsize=13,
                weight='bold'
            )
        
        ax.set_title(
            '🏙️ City People Density Heatmap\n',
            fontsize=18,
            weight='bold',
            pad=20
        )
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if self.config.save_plots and timestamp:
            save_path = Path(self.config.results_dir) / f"heatmap_{timestamp}.png"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"💾 Heatmap saved: {save_path}")
        
        plt.show()
    
    def plot_statistics(self, stats: List[DistrictStats], timestamp: str = None):
        """Create comprehensive statistics dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        names = [s.name for s in stats]
        colors = plt.cm.viridis(np.linspace(0, 1, len(stats)))
        
        # 1. Average Density
        ax1 = fig.add_subplot(gs[0, 0])
        avg_people = [s.avg_people for s in stats]
        bars = ax1.bar(names, avg_people, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('📊 Average Density', fontsize=14, weight='bold')
        ax1.set_ylabel('People per Frame', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 2. Peak Density
        ax2 = fig.add_subplot(gs[0, 1])
        max_people = [s.max_people for s in stats]
        bars = ax2.bar(names, max_people, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('📈 Peak Density', fontsize=14, weight='bold')
        ax2.set_ylabel('Maximum People', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Total Detections
        ax3 = fig.add_subplot(gs[0, 2])
        total_people = [s.total_people for s in stats]
        bars = ax3.bar(names, total_people, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_title('👥 Total Detections', fontsize=14, weight='bold')
        ax3.set_ylabel('Total People', fontsize=11)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # 4. Density Variation
        ax4 = fig.add_subplot(gs[1, 0])
        std_people = [s.std_people for s in stats]
        ax4.bar(names, std_people, color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_title('📉 Density Variation', fontsize=14, weight='bold')
        ax4.set_ylabel('Standard Deviation', fontsize=11)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Processing Efficiency
        ax5 = fig.add_subplot(gs[1, 1])
        processing_times = [s.processing_time for s in stats]
        ax5.bar(names, processing_times, color=colors, edgecolor='black', linewidth=1.5)
        ax5.set_title('⚡ Processing Time', fontsize=14, weight='bold')
        ax5.set_ylabel('Seconds', fontsize=11)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Average Confidence
        ax6 = fig.add_subplot(gs[1, 2])
        confidences = [s.avg_confidence * 100 for s in stats]
        ax6.bar(names, confidences, color=colors, edgecolor='black', linewidth=1.5)
        ax6.set_title('🎯 Detection Confidence', fontsize=14, weight='bold')
        ax6.set_ylabel('Average Confidence (%)', fontsize=11)
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim([0, 100])
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Density Distribution
        ax7 = fig.add_subplot(gs[2, :2])
        data_for_box = [s.frame_counts for s in stats]
        bp = ax7.boxplot(data_for_box, labels=names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax7.set_title('📦 Density Distribution', fontsize=14, weight='bold')
        ax7.set_ylabel('People per Frame', fontsize=11)
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Density Categories
        ax8 = fig.add_subplot(gs[2, 2])
        categories = [s.get_density_category() for s in stats]
        category_counts = pd.Series(categories).value_counts()
        ax8.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3(range(len(category_counts)))
        )
        ax8.set_title('🏷️ Density Categories', fontsize=14, weight='bold')
        
        fig.suptitle('📊 City Density Analysis Dashboard', fontsize=20, weight='bold', y=0.98)
        
        if self.config.save_plots and timestamp:
            save_path = Path(self.config.results_dir) / f"dashboard_{timestamp}.png"
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"💾 Dashboard saved: {save_path}")
        
        plt.show()
    
    def print_summary(self, stats: List[DistrictStats]):
        """Print beautiful summary table to console."""
        table = Table(title="🏙️  City Density Analysis Summary", show_header=True, header_style="bold cyan")
        
        table.add_column("District", style="bold yellow", width=15)
        table.add_column("Total", justify="right", style="green")
        table.add_column("Avg/Frame", justify="right", style="cyan")
        table.add_column("Max", justify="right", style="red")
        table.add_column("Std Dev", justify="right", style="magenta")
        table.add_column("Category", style="bold blue")
        table.add_column("Confidence", justify="right", style="white")
        
        for s in stats:
            table.add_row(
                s.name,
                f"{s.total_people:,}",
                f"{s.avg_people:.2f}",
                str(s.max_people),
                f"{s.std_people:.2f}",
                s.get_density_category(),
                f"{s.avg_confidence*100:.1f}%"
            )
        
        # Add summary row
        total = sum(s.total_people for s in stats)
        avg = np.mean([s.avg_people for s in stats]) if stats else 0
        
        table.add_section()
        table.add_row(
            "[bold]CITY TOTAL[/bold]",
            f"[bold]{total:,}[/bold]",
            f"[bold]{avg:.2f}[/bold]",
            "", "", "", "",
            style="bold white on blue"
        )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

# =====================================================
# CLI INTERFACE
# =====================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="🏙️  City People Density Analyzer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['fast', 'balanced', 'accurate'],
        default='balanced',
        help='Processing mode (default: balanced)'
    )
    
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show live preview during processing'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()

# =====================================================
# MAIN APPLICATION
# =====================================================

def create_default_config() -> AppConfig:
    """Create default configuration."""
    return AppConfig(
        model=ModelConfig(
            path="yolov8n.pt",
            confidence_threshold=0.3,
            use_gpu=True
        ),
        processing=ProcessingConfig(
            mode="balanced"
        ),
        output=OutputConfig(
            results_dir="results",
            export_formats=["json", "csv"]
        ),
        districts=[
            DistrictConfig(
                name="District A",
                video_path="videos/A_trialvideo.mp4",
                position=(0, 0)
            ),
            DistrictConfig(
                name="District B",
                video_path="videos/B_trialvideo.mp4",
                position=(0, 1)
            ),
            DistrictConfig(
                name="District C",
                video_path="videos/C_trialvideo.mp4",
                position=(1, 0)
            ),
            DistrictConfig(
                name="District D",
                video_path="videos/D_trialvideo.mp4",
                position=(1, 1)
            ),
        ],
        grid_shape=(2, 2)
    )

def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    console = Console()
    
    try:
        # Load configuration
        if args.config:
            console.print(f"[cyan]Loading configuration from: {args.config}[/cyan]")
            if args.config.endswith(('.yaml', '.yml')):
                config = AppConfig.from_yaml(args.config)
            elif args.config.endswith('.json'):
                config = AppConfig.from_json(args.config)
            else:
                raise ValueError("Config file must be YAML or JSON")
        else:
            console.print("[cyan]Using default configuration[/cyan]")
            config = create_default_config()
        
        # Override with CLI arguments
        if args.mode:
            config.processing.mode = args.mode
        if args.preview:
            config.processing.show_preview = True
        if args.output:
            config.output.results_dir = args.output
        if args.no_plots:
            config.output.save_plots = False
        
        # Save config
        config_save_path = Path(config.output.results_dir) / "last_run_config.yaml"
        config.save_yaml(str(config_save_path))
        logger.info(f"Configuration saved to: {config_save_path}")
        
        # Initialize analyzer
        analyzer = CityAnalyzer(config, logger)
        
        # Run analysis
        start_time = datetime.now()
        heatmap, stats = analyzer.analyze_city()
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Save results
        output_files = analyzer.save_results(heatmap, stats)
        
        # Visualize
        visualizer = Visualizer(config.output, logger)
        visualizer.print_summary(stats)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if config.output.save_plots:
            visualizer.plot_heatmap(heatmap, config.districts, timestamp)
            visualizer.plot_statistics(stats, timestamp)
        
        # Final summary
        console.print(Panel.fit(
            f"[bold green]✅ Analysis completed successfully![/bold green]\n\n"
            f"⏱️  Total processing time: {total_time:.1f}s\n"
            f"📊 Districts analyzed: {len(stats)}\n"
            f"👥 Total people detected: {sum(s.total_people for s in stats):,}\n"
            f"💾 Results saved to: {config.output.results_dir}",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
        logger.warning("Analysis interrupted")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        logger.exception("Application error")
        sys.exit(1)

if __name__ == "__main__":
    main()