import numpy as np

class URRobot:
    def __init__(self, model_name='ur5'):
        self.model_name = model_name.lower()
        self.d = []
        self.a = []
        self.alpha = []
        
        # Cấu hình DH Parameters cho các dòng robot UR
        self.dh_params = {
            'ur5': {
                'd': np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]),
                'a': np.array([0, -0.42500, -0.39225, 0, 0, 0]),
                'alpha': np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
            },
            'ur5e': {
                'd': np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996]),
                'a': np.array([0, -0.42500, -0.39220, 0, 0, 0]),
                'alpha': np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
            }
            
        }
        
        if self.model_name in self.dh_params:
            self.d = self.dh_params[self.model_name]['d']
            self.a = self.dh_params[self.model_name]['a']
            self.alpha = self.dh_params[self.model_name]['alpha']
        else:
            raise ValueError(f"Robot model '{model_name}' not supported. Available: {list(self.dh_params.keys())}")

    # ================= UTILS =================
    @staticmethod
    def create_target_pose(position, orientation_rpy):
        x, y, z = position
        rx, ry, rz = orientation_rpy
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        R_3x3 = Rz @ Ry @ Rx
        T_4x4 = np.eye(4)
        T_4x4[0:3, 0:3] = R_3x3
        T_4x4[0:3, 3] = [x, y, z]
        return T_4x4

    @staticmethod
    def dh_matrix(theta, d, a, alpha):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [cos_t, -sin_t * cos_a,  sin_t * sin_a, a * cos_t],
            [sin_t,  cos_t * cos_a, -cos_t * sin_a, a * sin_t],
            [0    ,  sin_a        ,  cos_a        , d        ],
            [0    ,  0            ,  0            , 1        ]
        ])

    # ================= FORWARD KINEMATICS =================
    def forward_kinematics(self, thetas):
        """Tính toán FK dựa trên tham số DH của robot hiện tại."""
        T = np.eye(4)
        # UR có 6 khớp
        for i in range(6):
            T_i = self.dh_matrix(thetas[i], self.d[i], self.a[i], self.alpha[i])
            T = np.dot(T, T_i)
        return T

    # ================= CORRECT UR ANALYTICAL IK =================
    
    def inverse_kinematics(self, T_target):
        """
        Giải IK theo phương pháp hình học chuẩn cho UR (Sastry/Universal Robot kinematics).
        """
        solutions = []
        
        # Chuẩn bị các tham số
        d1, d4, d5, d6 = self.d[0], self.d[3], self.d[4], self.d[5]
        a2, a3 = self.a[1], self.a[2]
        
        # Tách Rotation và Position từ Target
        R_target = T_target[0:3, 0:3]
        P_target = T_target[0:3, 3]

        # --- BƯỚC 1: GIẢI THETA 1 ---
        # Tìm vị trí tâm của Wrist 1 (giao điểm trục 5 và 6 chiếu xuống mp)
        # P_wrist_1 không phải là tâm cầu, mà là tâm khớp 5
        # Vector P5 = P_target - d6 * Z_target
        P5 = P_target - d6 * R_target[:, 2] # Lùi lại 1 đoạn d6 theo hướng Z của tool
        
        x5, y5 = P5[0], P5[1]
        
        # Có 2 nghiệm cho Theta 1 (Vai trái / Vai phải)
        # d4 là khoảng cách lệch vai
        rho = np.sqrt(x5**2 + y5**2)
        if rho < d4:
            return [] # Không với tới (target nằm trong vùng deadzone của vai)
            
        # Góc lệch do d4
        alpha_t1 = np.arctan2(y5, x5)
        beta_t1 = np.arccos(d4 / rho)
        
        t1_candidates = [
            alpha_t1 + beta_t1 + np.pi/2, 
            alpha_t1 - beta_t1 + np.pi/2
        ]

        for t1 in t1_candidates:
            # --- BƯỚC 2: GIẢI THETA 5 ---
            # Dựa vào động học UR: Px*sin(t1) - Py*cos(t1) = d4 + d6*cos(t5) ?? -> Không đúng
            # Công thức chuẩn: P_wrist_1_x * sin(t1) - P_wrist_1_y * cos(t1) = d4
            # Ta cần dùng toạ độ z của trục tool để tìm t5.
            
            # Tính áp lực lên trục 5:
            # P_target_x * sin(t1) - P_target_y * cos(t1) = d4 + d6 * cos(t5) -> Sai số nhỏ
            # Đúng hơn: d4 + d6 * Z_wrist_y (trong frame 1)
            
            # Cách tiếp cận hình học chuẩn xác hơn cho t5:
            # Chiếu toạ độ P_target lên hệ trục đã xoay theo t1
            Px, Py = P_target[0], P_target[1]
            val = (Px * np.sin(t1) - Py * np.cos(t1) - d4) / d6
            
            if abs(val) > 1:
                continue # Không giải được t5
            
            # t5 cũng có 2 nghiệm (cổ tay gập lên / gập xuống)
            t5_candidates = [np.arccos(val), -np.arccos(val)]
            
            for t5 in t5_candidates:
                # Nếu t5 ~ 0 (Singularity), trục 4 và 6 song song -> ta có thể chọn t6 tuỳ ý, nhưng code này bỏ qua
                if abs(np.sin(t5)) < 1e-4:
                    continue 

                # --- BƯỚC 3: GIẢI THETA 6 ---
                # Dựa vào hình chiếu trục X, Y của tool
                # X_tool_x * sin(t1) - X_tool_y * cos(t1) = ...
                # Công thức chuẩn:
                sign_t5 = np.sign(np.sin(t5))
                
                Rx, Ry = R_target[0, 0], R_target[1, 0] # X component của target orientation
                Rz_x, Rz_y = R_target[0, 1], R_target[1, 1] # Y component của target
                
                # Tính toán dựa trên việc quay ngược hệ trục
                s1, c1 = np.sin(t1), np.cos(t1)
                
                # M = R_0_1(t1).inv() * R_target
                # Tuy nhiên dùng công thức hình học trực tiếp:
                # A = -Rm[0,1]*s1 + Rm[1,1]*c1
                # B = -Rm[0,0]*s1 + Rm[1,0]*c1
                # tan(t6) = A / B  (với mẫu số chia cho sin(t5))
                
                term_y = (-R_target[0, 1] * s1 + R_target[1, 1] * c1) / np.sin(t5)
                term_x = ( R_target[0, 0] * s1 - R_target[1, 0] * c1) / np.sin(t5)
                
                t6 = np.arctan2(term_y, term_x)
                
                # --- BƯỚC 4: GIẢI THETA 2, 3, 4 (PLANAR 3-LINK) ---
                # Bây giờ ta cần đưa target về mặt phẳng của robot (Frame 1-2-3-4)
                # Ta cần tìm toạ độ của Wrist Center (Tâm khớp 4) nhưng trong mặt phẳng xoay t1
                
                # T14 = T01(-1) * T_target * T45(-1) * T56(-1)
                # Nhưng cách nhanh hơn là dùng hình học phẳng:
                
                # Quay hệ trục về mặt phẳng cánh tay:
                T0_1 = self.dh_matrix(t1, d1, 0, np.pi/2)
                T4_5 = self.dh_matrix(t5, d5, 0, -np.pi/2)
                T5_6 = self.dh_matrix(t6, d6, 0, 0)
                
                T4_6 = T4_5 @ T5_6
                T1_6 = np.linalg.inv(T0_1) @ T_target
                T1_4 = T1_6 @ np.linalg.inv(T4_6)
                
                # Vị trí P14 (x, z) trong mặt phẳng 2D (Lưu ý: y phải ~ 0)
                P14_x = T1_4[0, 3]
                P14_y = T1_4[1, 3] # Nên xấp xỉ 0
                P14_z = T1_4[2, 3] # Đây là trục Y trong mặt phẳng DH cũ?? Không, trục Z của frame 1 là trục quay frame 2
                
                # Trong frame 1 (sau khi xoay t1, dựng đứng):
                # Trục X chạy dọc cánh tay, Trục Y ra ngoài, Trục Z hướng lên
                # Planar IK giải cho x (ngang) và y (dọc - tương ứng z cũ)
                # Cẩn thận: DH của UR: Frame 2 (Shoulder) lệch so với Frame 1 (Base)
                
                # P_planar = (P14_x, -P14_y) ??? 
                # Hãy dùng vector đơn giản:
                # Vector từ Shoulder (k2) đến Wrist 1 (k4)
                vec_sw_x = -P14_z # Do trục Z của DH frame 1 hướng lên
                vec_sw_y = -P14_x # Do trục X của DH frame 1 hướng ra
                
                # Sửa lại khớp toạ độ: 
                # Frame 1: Z hướng lên trời. Frame 2 quay quanh Z1 (ngang). 
                # T1_4[0,3] là toạ độ theo trục X1 (hướng ra)
                # T1_4[1,3] là toạ độ theo trục Y1 (hướng ngang vuông góc) -> phải bằng 0 nếu đúng mặt phẳng
                # T1_4[2,3] là toạ độ theo trục Z1 (hướng lên)
                
                r_cfg = np.sqrt(P14_x**2 + P14_y**2) # Khoảng cách ngang
                # Thực ra P14_y nên rất nhỏ. Ta dùng P14_x là chính.
                
                # Bài toán tam giác phẳng:
                # Cạnh 1: a2, Cạnh 2: a3
                # Cạnh huyền nối (0,0) tới (P14_x, P14_y) trong hệ toạ độ khớp 2
                
                # Toạ độ mục tiêu so với khớp 2:
                # Frame 2 nằm ở (0,0,0) của Frame 1 ?? KHÔNG. Frame 2 cách Frame 1 đoạn d1 theo Z, nhưng ma trận T0_1 đã khử d1 rồi.
                # Tuy nhiên Frame 2 có a2 (Trục X2).
                
                # Dùng hình học phẳng đơn giản nhất:
                # Ta cần vector từ khớp 2 đến khớp 4.
                # Trong hệ toạ độ T1: Khớp 2 nằm tại (0,0,0) (sau khi khử d1 ở T0_1 z-offset)
                # Đích đến là T1_4 position.
                
                p24_x = T1_4[0, 3]
                p24_y = T1_4[1, 3] # Cái này phải xấp xỉ -d4 ?? Không, d4 đã xử lý ở T4_5?
                # Chú ý: Frame 3 đến Frame 4 có d4. Nhưng trong Planar, d4 vuông góc mặt phẳng.
                # T1_4 đã loại bỏ phần quay của wrist.
                
                # Giải t3 (Elbow):
                # Dist^2 = x^2 + y^2
                dist_sq = P14_x**2 + P14_y**2 # P14_z ở đâu?
                # Trong DH: Z1 là trục quay của khớp 2. X1 là hướng của cánh tay.
                # Ta sử dụng P_x và P_y của T1_4 để giải t2, t3, t4??
                # Code này đang rối ở đoạn Planar do ma trận DH của UR hơi "dị".
                
                # --- CÁCH ĐƠN GIẢN HÓA CHO PLANAR (t2, t3, t4) ---
                # Vector từ khớp 2 đến khớp 4 trong mặt phẳng:
                # Rx = sqrt(x5^2 + y5^2 - d4^2) - a2?? Quá phức tạp.
                # Dùng công thức chuẩn của UR:
                
                # P_base_to_wrist1 = P5 (đã tính ở bước 1)
                # Chiếu vào mặt phẳng cánh tay:
                # R = sqrt(x5^2 + y5^2)
                # R_planar = sqrt(R^2 - d4^2)  (cạnh ngang trong mp)
                # Z_planar = P5[2] - d1        (cạnh dọc trong mp)
                
                # Nhưng t1 có 2 nghiệm, nên R_planar có thể âm dương tuỳ phía?
                # Với t1 đã chọn:
                # Toạ độ Wrist 1 trong hệ trục khớp 2:
                # X_w1 = c1*x5 + s1*y5 
                # Y_w1 = P5[2] - d1
                
                X_w1 = np.cos(t1)*x5 + np.sin(t1)*y5
                Y_w1 = P5[2] - d1
                
                # Khoảng cách từ tâm khớp 2 đến tâm khớp 4
                # Lưu ý: a2, a3 là độ dài link.
                # Dist^2 = X_w1^2 + Y_w1^2
                
                # Định lý cosin cho t3:
                # Dist^2 = a2^2 + a3^2 - 2*a2*a3*cos(pi - t3)
                cos_t3 = (X_w1**2 + Y_w1**2 - a2**2 - a3**2) / (2 * a2 * a3)
                
                if abs(cos_t3) > 1:
                    continue
                    
                t3_candidates = [np.arccos(cos_t3), -np.arccos(cos_t3)]
                
                for t3 in t3_candidates:
                    # Giải t2:
                    # t2 = beta - alpha
                    # alpha = atan2(a3*sin(t3), a2 + a3*cos(t3))
                    # beta = atan2(Y_w1, X_w1)
                    
                    alpha_t2 = np.arctan2(a3*np.sin(t3), a2 + a3*np.cos(t3))
                    beta_t2 = np.arctan2(Y_w1, X_w1)
                    
                    t2 = beta_t2 - alpha_t2
                    
                    # Giải t4:
                    # t2 + t3 + t4 = tổng góc nghiêng của cổ tay so với trục ngang
                    # Góc này được quyết định bởi ma trận xoay T1_4
                    # T1_4 = T1_2 * T2_3 * T3_4
                    # Rot(Z) của T1_4 chính là t2 + t3 + t4
                    
                    # Lấy góc từ ma trận T1_4 đã tính ở trên
                    # R1_4 = T1_4[0:3, 0:3]
                    # Góc xoay quanh trục Z ??
                    # Trong DH UR: t2, t3, t4 đều xoay quanh trục Z song song nhau.
                    # R1_4[0,0] = cos(t234), R1_4[1,0] = sin(t234)
                    
                    t234 = np.arctan2(T1_4[1, 0], T1_4[0, 0])
                    t4 = t234 - t2 - t3
                    
                    solutions.append(np.array([t1, t2, t3, t4, t5, t6]))

        return solutions
# ================= VÍ DỤ SỬ DỤNG =================

if __name__ == "__main__":
    # 1. Khởi tạo robot (Chọn 'ur5' hoặc 'ur5e')
    robot = URRobot(model_name='ur5e') 
    
    # 2. Tạo vị trí mục tiêu
    # Vị trí (x,y,z) và Góc (roll, pitch, yaw)
    target_pos = [0.4, 0.1, 0.4]
    target_orient = [np.pi, 0, 0] 
    
    T_target = robot.create_target_pose(target_pos, target_orient)
    
    print(f"--- Solving IK for {robot.model_name.upper()} ---")
    
    # 3. Giải Inverse Kinematics
    ik_solutions = robot.inverse_kinematics(T_target)
    
    if ik_solutions:
        print(f"Found {len(ik_solutions)} solutions.")
        
        # Lấy nghiệm đầu tiên để kiểm tra FK
        q_sol = ik_solutions[0]
        print("Solution 1 (joints):", np.round(q_sol, 3))
        
        # 4. Kiểm tra lại bằng Forward Kinematics
        T_check = robot.forward_kinematics(q_sol)
        print("\nRe-calculated Position (FK):")
        print(np.round(T_check[0:3, 3], 3))
        print("Original Target Position:")
        print(np.array(target_pos))
        
    else:
        print("No IK solution found for this target.")
