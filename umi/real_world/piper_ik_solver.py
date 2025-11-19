#!/usr/bin/env python3
"""
Standalone Piper IK Solver using Pinocchio
Adapted from AgileX PikaAnyArm implementation
Removed ROS dependencies for use in UMI
"""
import casadi
import math
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import os
from pathlib import Path

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def matrix_to_xyzrpy(matrix):
    """Convert 4x4 transformation matrix to [x, y, z, roll, pitch, yaw]"""
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    """Create 4x4 transformation matrix from position and RPY angles"""
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    return transformation_matrix


class PiperIKSolver:
    """
    Pinocchio-based IK solver for Piper robot
    Uses optimization to find smooth, collision-free joint configurations
    """
    def __init__(self, tool_offset=0.182, verbose=False):
        """
        Args:
            tool_offset: TCP offset from J6 flange (meters)
            verbose: Enable debug printing
        """
        self.verbose = verbose
        self.tool_offset = tool_offset
        
        # Find URDF file
        urdf_path = self._find_urdf()
        if not urdf_path:
            raise FileNotFoundError("Could not find piper URDF file")
        
        if self.verbose:
            print(f"[PiperIKSolver] Loading URDF from: {urdf_path}")
        
        # Build robot model
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path))
        
        # Lock gripper joints (joint7, joint8)
        mixed_joints_to_lock = ["joint7", "joint8"]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=mixed_joints_to_lock,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )
        
        # Define TCP frame with tool offset
        # First rotation: -90deg around Y axis
        first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        # Second: tool offset along X
        second_matrix = create_transformation_matrix(tool_offset, 0, 0, 0, 0, 0)
        self.tcp_transform = np.dot(first_matrix, second_matrix)
        
        # Add TCP frame to model
        from scipy.spatial.transform import Rotation as R
        rot_mat = self.tcp_transform[:3, :3]
        quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        
        self.reduced_robot.model.addFrame(
            pin.Frame('tcp',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          pin.Quaternion(quat[3], quat[0], quat[1], quat[2]),  # w, x, y, z
                          np.array([self.tcp_transform[0, 3], self.tcp_transform[1, 3], self.tcp_transform[2, 3]]),
                      ),
                      pin.FrameType.OP_FRAME)
        )
        
        # Initialize state tracking
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)
        
        # Setup CasADi optimization
        self._setup_optimization()
        
        if self.verbose:
            print(f"[PiperIKSolver] Initialized with {self.reduced_robot.model.nq} DOF")
    
    def _find_urdf(self):
        """Find Piper URDF file in common locations"""
        # Try common locations
        search_paths = [
            Path(__file__).parent.parent.parent / "PikaAnyArm/piper/piper_description/urdf/piper_description.urdf",
            Path.home() / "universal_manipulation_interface/PikaAnyArm/piper/piper_description/urdf/piper_description.urdf",
            Path("/opt/ros/noetic/share/piper_description/urdf/piper_description.urdf"),
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _setup_optimization(self):
        """Setup CasADi optimization problem for IK"""
        # Create CasADi model
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        
        # Symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # Get TCP frame ID
        self.tcp_id = self.reduced_robot.model.getFrameId("tcp")
        
        # Define error function (SE3 log)
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.tcp_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )
        
        # Setup optimization
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        
        # Cost function with weighted position and orientation
        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]
        weight_position = 1.0
        weight_orientation = 0.1
        
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        self.regularization = casadi.sumsqr(self.var_q)
        
        # Joint limits
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        
        # Minimize cost
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        
        # Solver options
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
    
    def solve_ik(self, target_pose_matrix, initial_q=None):
        """
        Solve IK for target TCP pose
        
        Args:
            target_pose_matrix: 4x4 homogeneous transformation matrix
            initial_q: Initial joint configuration (optional)
        
        Returns:
            (joint_angles, success): Tuple of 6D joint angles and success flag
        """
        if initial_q is not None:
            self.init_data = initial_q
        
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, target_pose_matrix)
        
        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)
            
            # Check for drastic changes
            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                self.init_data = sol_q
                
                if max_diff > 30.0 / 180.0 * 3.1415:
                    if self.verbose:
                        print(f"[PiperIKSolver] Large joint change: {max_diff:.3f} rad")
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            
            self.history_data = sol_q
            
            # Verify solution accuracy
            achieved_pose = self.get_tcp_pose(sol_q)
            target_xyz = matrix_to_xyzrpy(target_pose_matrix)
            
            diff_pos = np.linalg.norm(np.array(achieved_pose[:3]) - np.array(target_xyz[:3]))
            
            if diff_pos > 0.01:  # 1cm tolerance
                if self.verbose:
                    print(f"[PiperIKSolver] IK solution error: {diff_pos:.4f}m")
                return sol_q, False
            
            return sol_q, True
            
        except Exception as e:
            if self.verbose:
                print(f"[PiperIKSolver] IK failed: {e}")
            return None, False
    
    def get_tcp_pose(self, q):
        """
        Get TCP pose from joint configuration
        
        Args:
            q: Joint angles (6D)
        
        Returns:
            [x, y, z, roll, pitch, yaw]
        """
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.array(q))
        
        # Get J6 pose
        j6_pose = self.reduced_robot.data.oMi[6]
        j6_matrix = j6_pose.homogeneous
        
        # Apply TCP transform
        tcp_matrix = np.dot(j6_matrix, self.tcp_transform)
        
        return matrix_to_xyzrpy(tcp_matrix)
