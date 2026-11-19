"""案例圖：預測方位線與 HOTSPOT 像素座標的幾何。"""
import os
import sys
import unittest

from _support import ROOT, temp_dir, write_json  # noqa: F401

_TOOL = os.path.join(ROOT, "stitch_tool")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

import gt_hotspot  # noqa: E402
from pose_regions.render_az_hotspot_case import az_to_x, band_x_ranges  # noqa: E402


class TestAzToX(unittest.TestCase):
    def test_image_center_is_zero_azimuth(self):
        self.assertAlmostEqual(az_to_x(0.0, 360), 180.0)

    def test_right_quarter_is_plus_90(self):
        self.assertAlmostEqual(az_to_x(90.0, 360), 270.0)

    def test_seam_maps_to_left_edge(self):
        self.assertAlmostEqual(az_to_x(180.0, 360), 0.0)
        self.assertAlmostEqual(az_to_x(-180.0, 360), 0.0)


class TestBandXRanges(unittest.TestCase):
    def test_band_around_center_is_one_segment(self):
        ranges = band_x_ranges(0.0, 22.5, 360)
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0][0], az_to_x(-22.5, 360))
        self.assertAlmostEqual(ranges[0][1], az_to_x(22.5, 360))

    def test_band_across_seam_splits_in_two(self):
        ranges = band_x_ranges(180.0, 22.5, 360)
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0][0], az_to_x(157.5, 360))
        self.assertAlmostEqual(ranges[0][1], 360.0)
        self.assertAlmostEqual(ranges[1][0], 0.0)
        self.assertAlmostEqual(ranges[1][1], az_to_x(202.5, 360))


class TestLoadGtLocations(unittest.TestCase):
    def test_keeps_pixel_xy_not_just_azimuth(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "a_HOTSPOT.json"), {
                "HOTSPOTOFROOM": [{
                    "IDName": "a.jpg",
                    "Roomidx": 0,
                    "ToIDName": {"IDName": ["b.jpg"]},
                    "HSLoc": {"Coordinate": [{"x": 285.0, "y": 1832.0}]},
                }],
            })
            loc = gt_hotspot.load_gt_locations(gt_hotspot.find_hotspot_files(d))
            self.assertEqual(loc["a.jpg"]["b.jpg"], (285.0, 1832.0))
