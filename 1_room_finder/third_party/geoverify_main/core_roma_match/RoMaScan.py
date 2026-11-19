import os
import cv2
import numpy as np
import torch
import tempfile
from sklearn.cluster import DBSCAN, KMeans
from PIL import Image

# Import everything from submodules to re-expose them
from core_roma_match.proection import ProjectionEngine, FOVOptimizer
from core_roma_match.clustering import DBSCANClusterManager, KMeansClusterManager
from core_roma_match.geometry_filter import GeometryFilter
from core_roma_match.matchers import MatcherBase, RomaV1Matcher, RomaV2Matcher, MatcherFactory
from core_roma_match.refiner import LocalRefiner
from core_roma_match.pipeline import ImageHandler, RoMaScanPipeline, RomaScan
from core_roma_match.visualization import (
    select_diverse_indices, conf_to_bgr, draw_confidence_colorbar, visualize_slice_matches
)

__all__ = [
    "ProjectionEngine", "FOVOptimizer",
    "DBSCANClusterManager", "KMeansClusterManager",
    "GeometryFilter",
    "MatcherBase", "RomaV1Matcher", "RomaV2Matcher", "MatcherFactory",
    "LocalRefiner",
    "ImageHandler", "RoMaScanPipeline", "RomaScan",
    "select_diverse_indices", "conf_to_bgr", "draw_confidence_colorbar", "visualize_slice_matches"
]

# depth 模組屬於評估路徑，不在本交付套件內；本檔後續未使用這兩個名稱，
# 保留為 None 以維持介面。需要深度估計時由呼叫端自行提供。
DepthAnythingEstimator = None
MidasEstimator = None
