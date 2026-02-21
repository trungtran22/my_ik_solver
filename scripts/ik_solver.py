import numpy as np

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

class URRobot:
    def __init__(self, model_name='ur5e'):
        self.model_name = model_name.lower()

        # Official-style DH tables (standard DH) from UR docs
        # (UR also has calibration deltas in real robots, but these are the nominal tables.)
        # Source: Universal Robots DH parameter article. :contentReference[oaicite:2]{index=2}
        self.dh_params = {
            'ur5': {
                'd': np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]),
                'a': np.array([0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0]),
                'alpha': np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])
            },
            'ur5e': {
                'd': np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]),
                'a': np.array([0.0, -0.42500, -0.39220, 0.0, 0.0, 0.0]),
                'alpha': np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])
            },
            'ur3e': {
                'd': np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]),
                'a': np.array([0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0]),
                'alpha': np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])
            }
        }

        if self.model_name not in self.dh_params:
            raise ValueError(f"Robot model '{model_name}' not supported. Available: {list(self.dh_params.keys())}")

        self.d = self.dh_params[self.model_name]['d']
        self.a = self.dh_params[self.model_name]['a']
        self.alpha = self.dh_params[self.model_name]['alpha']

    @staticmethod
    def create_target_pose(position, orientation_rpy):
        x, y, z = position
        rx, ry, rz = orientation_rpy

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def dh_matrix(theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,      sa,     ca,    d],
            [0,       0,      0,    1]
        ])

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i in range(6):
            T = T @ self.dh_matrix(q[i], self.d[i], self.a[i], self.alpha[i])
        return T

    def _R03_from_q123(self, q1, q2, q3):
        T = np.eye(4)
        for i, qi in enumerate([q1, q2, q3]):
            T = T @ self.dh_matrix(qi, self.d[i], self.a[i], self.alpha[i])
        return T[:3, :3]

    def inverse_kinematics(self, T06, eps=1e-9):
        """
        Analytic IK (nominal DH). Returns list of solutions [q1..q6] in radians.
        """
        d1, d4, d5, d6 = self.d[0], self.d[3], self.d[4], self.d[5]
        a2, a3 = self.a[1], self.a[2]

        R06 = T06[:3, :3]
        p06 = T06[:3, 3]

        # p05 = p06 - d6 * z06  (frame-5 origin)
        p05 = p06 - d6 * R06[:, 2]
        x, y, z = p05

        # ---- Solve q1 (2 solutions) ----
        r = np.hypot(x, y)
        if r < abs(d4) - 1e-12:
            return []  # unreachable

        # shoulder angle geometry
        phi = np.arctan2(y, x)
        gamma = np.arccos(np.clip(d4 / r, -1.0, 1.0))

        q1_list = [phi + gamma + np.pi/2, phi - gamma + np.pi/2]

        sols = []

        for q1 in q1_list:
            s1, c1 = np.sin(q1), np.cos(q1)

            # project wrist-center into the arm plane after q1
            # common UR trick: x1 = c1*x + s1*y  (distance along arm direction)
            x1 = c1*x + s1*y
            z1 = z - d1

            # Because d4 is the offset from joint3->joint4, it sits perpendicular to the arm plane.
            # The in-plane distance to the "elbow triangle" uses x1 and z1 only.
            D = (x1**2 + z1**2 - a2**2 - a3**2) / (2*a2*a3)

            if abs(D) > 1.0:
                continue

            q3_candidates = [np.arccos(np.clip(D, -1.0, 1.0)),
                             -np.arccos(np.clip(D, -1.0, 1.0))]

            for q3 in q3_candidates:
                # Solve q2
                k1 = a2 + a3*np.cos(q3)
                k2 = a3*np.sin(q3)
                q2 = np.arctan2(z1, x1) - np.arctan2(k2, k1)

                # Now solve wrist from rotation
                R03 = self._R03_from_q123(q1, q2, q3)
                R36 = R03.T @ R06

                # q5 from R36[2,2]
                # Using: q5 = atan2( sqrt(r13^2+r23^2), r33 )
                s5 = np.hypot(R36[0, 2], R36[1, 2])
                q5a = np.arctan2(s5, R36[2, 2])
                q5b = np.arctan2(-s5, R36[2, 2])  # second branch

                for q5 in [q5a, q5b]:
                    if abs(np.sin(q5)) < 1e-6:
                        # wrist singularity: q4 and q6 coupled
                        # choose q4 = 0, solve q6 from remaining yaw
                        q4 = 0.0
                        q6 = np.arctan2(-R36[1, 0], R36[0, 0])
                    else:
                        q4 = np.arctan2(R36[1, 2]/np.sin(q5), R36[0, 2]/np.sin(q5))
                        q6 = np.arctan2(R36[2, 1]/np.sin(q5), -R36[2, 0]/np.sin(q5))

                    q = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
                    q = np.array([wrap_to_pi(v) for v in q])
                    sols.append(q)

        # optional: remove near-duplicates
        uniq = []
        for q in sols:
            if not any(np.allclose(q, u, atol=1e-5) for u in uniq):
                uniq.append(q)
        return uniq
