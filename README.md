# ik_solver for universal robot
## Inverse Kinematics Solver and Motion Planning RRT for UR5
### Test on ROS 1 Noetic - Python 3
**IK idea: Kinematics Decoupling**
- First, calculate $$\theta_1, \theta_2, \theta_3$$, move wrist center to correct target position
- Then, calculate $$\theta_4, \theta_5, \theta_6$$ from current to target orientation
1. Calculate $P_c$ (wrist center) from target transformation matrix $T^{0}_{6}$
   - $$P_c = P_e - d6*$$(orientation z of the end effector)
2. Calculate $\theta_1$: 2 solutions
   - $$\theta_1 = \text{atan2}(y_c, x_c) \pm \arccos\left(\frac{d_4}{\sqrt{x_c^2 + y_c^2}}\right)$$
3. First loop: find solutions of $$\theta_3, \theta_2$$ for each $\theta_1$
   1. Calculate $\theta_3$: 2 solutions; using Law of Cosine
      - $s = x_c \cos(\theta_1) + y_c \sin(\theta_1)$
      - $h = z_c - d_1$
      - $D = \sqrt{s^2 + h^2}$.
      - $$D^2 = a_2^2 + a_3^2 + 2 a_2 a_3 \cos(\theta_3)$$
      - $$\cos(\theta_3) = \frac{D^2 - a_2^2 - a_3^2}{2 a_2 a_3}$$
   2. Calculate $\theta_2$: each $\theta_3$ and $\theta_1$ has 1 $\theta_2$
      - $\theta_2 = \text{atan2}(h, s) - \text{atan2}(a_3 \sin(\theta_3), a_2 + a_3 \cos(\theta_3))$
   3. Calculate $T^{3}_{6}$
      - $$M = ((T^{0}_{3})^{-1}) * T^{0}_{6}$$
   4. Calculate $\theta_5$: 2 solutions
      - $cos(\theta_5)$ = $m_{33}$ from $M$
   5. Second loop:
      1. Calculate $\theta_4$:
         - $$\theta_4 = \text{atan2}\left( \frac{-m_{23}}{\sin(\theta_5)}, \frac{-m_{13}}{\sin(\theta_5)} \right)$$
         - if  $\sin(\theta_5) > 0$  (from $\theta_5$ > 0): $$\theta_4 = \text{atan2}(-m_{23}, -m_{13})$$
         - else: $$\theta_4 = \text{atan2}(m_{23}, m_{13})$$
      2. Calculate $\theta_6$:
         - $$\theta_6 = \text{atan2}\left( \frac{m_{32}}{\sin(\theta_5)}, \frac{m_{31}}{\sin(\theta_5)} \right)$$
         - $$\theta_6 = \text{atan2}(m_{32}, m_{31})$$ ($\sin(\theta_5) > 0$)
         - $$\theta_6 = \text{atan2}(-m_{32}, -m_{31})$$ ($\sin(\theta_5) < 0$)
      3. Save results: 8 solutions

### Init UR5 model:
- Init Moveit Planning and RViz:
  ```
  roslaunch ur5_moveit_config demo.launch
  ```
- Init UR5 inverse kinematics node and starting pick and place:
  ```
  rosrun ik_solver_ur5 main_ik.py
  ```
**Result**: 
- Performing Pick and Place with the IK Solver and Moveit Planning.
- It's still not working nicely with the orientation, i'm trying to fix on that.
- This result has not performed the RRT yet, only IK Solver.\
\
![](https://github.com/trungtran22/ik_solver_ur5/blob/main/pics/IMG_6565.GIF) 
