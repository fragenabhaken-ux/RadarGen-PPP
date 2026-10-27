"""The raw and raster producers must observe identical pre-dedup targets."""
import numpy as np

from radargen.radar_maps import creator


def test_raw_targets_match_raster_callback(monkeypatch):
    points = np.array([[1, 2, 0, 0, 0, 0, 3, 4],
                       [1, 2, 0, 0, 0, 0, 5, 6],
                       [50, 0, 0, 0, 0, 0, 7, 8]], dtype=np.float32)

    class Adapter:
        reference_sensor = 'REF'
        normalization = type('Normalization', (), {'rcs_min': -20, 'rcs_max': 66})()

        def load_radar_pointcloud(self, frame, nsweeps):
            return points.copy()

        def get_ref_to_vehicle_flat_up_transform(self, token):
            return np.eye(4)

        def filter_by_camera_frustum(self, pcl, frame, **kwargs):
            return pcl

    # Raster kernels are irrelevant to the extraction contract under test.
    for name in ('create_radar_pd_map', 'create_radar_rcs_map', 'create_radar_doppler_map'):
        monkeypatch.setattr(creator, name, lambda *args, **kwargs: np.zeros((8, 8)))
    adapter, frame = Adapter(), {'REF': 'reference'}
    captured = []
    creator.create_radar_maps_for_frame(adapter, frame, 'cpu', image_size=8,
                                       radar_callback=captured.append)
    raw = creator.load_filtered_radar_detections(adapter, frame)
    np.testing.assert_array_equal(raw, captured[0])
    np.testing.assert_array_equal(raw, points[:2])
