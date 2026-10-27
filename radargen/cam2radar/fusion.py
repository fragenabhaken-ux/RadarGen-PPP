"""Channel readout for camera grids after joint multi-view attention."""
from torch import nn


class CameraChannelFusion(nn.Module):
    """Project camera-major (B,V,D,Hp,Wp) to (B,D,Hp,Wp).

    Camera/feature axes are concatenated, not the spatial axes. This avoids
    tiling seams without claiming that equal patch indices are correspondences.
    The upstream joint neck establishes multi-view context before this readout.
    """

    def __init__(self, dim=384, num_cameras=4):
        super().__init__()
        self.num_cameras = num_cameras
        self.dim = dim
        self.projection = nn.Sequential(
            nn.Conv2d(num_cameras * dim, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU())

    def forward(self, camera_features):
        if camera_features.ndim != 5 or camera_features.shape[1:3] != (self.num_cameras, self.dim):
            raise ValueError('Expected camera-major (B,V,D,Hp,Wp) features')
        return self.projection(camera_features.flatten(1, 2))
