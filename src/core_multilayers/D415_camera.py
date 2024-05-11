import numpy as np

class CameraMgr():
    def __init__(self):
        self.intrinsic_rgb = np.array([[910.813728, 0, 640],
                              [0, 910.813728, 360],
                              [0, 0, 1]])
        self.intrinsic_ir = np.array([[896.866, 0, 640],
                             [0, 896.866, 360],
                             [0, 0, 1]])

        self.pose_rgb = np.eye(4, dtype=np.float32)
        self.pose_ir1 = np.array([[1., 0.0, 0.0, -0.0151],
                             [0.0, 1., 0.0, 0.0],
                             [0.0, 0.0, 1., 0.0],
                             [0, 0, 0, 1]])
        self.pose_ir2 = np.array([[1., 0.0, 0.0, -0.0701],
                             [0.0, 1., 0.0, 0.0],
                             [0.0, 0.0, 1., 0.0],
                             [0, 0, 0, 1]])

    def sim_d415_rgb(self):
        intrinsic = self.intrinsic_rgb.copy()
        intrinsic[: 2] *= 0.5
        intrinsic[0:2, 2] -= 0.5
        return intrinsic

    def sim_d415_ir(self):
        intrinsic = self.intrinsic_ir.copy()
        intrinsic[: 2] *= 0.5
        intrinsic[0:2, 2] -= 0.5
        return intrinsic

    def sim_ir12rgb(self):
        return np.linalg.inv(self.pose_ir1)

    def sim_d415_rgb_pose(self):
        return self.pose_rgb.copy()

    def sim_d415_ir1_pose(self):
        return self.pose_ir1.copy()

    def sim_d415_ir2_pose(self):
        return self.pose_ir2.copy()

