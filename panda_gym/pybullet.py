import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import panda_gym.assets


class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render (bool, optional): Enable rendering. Defaults to False.
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
    """

    def __init__(self, render: bool = False, n_substeps: int = 20,
                 background_color: Optional[np.ndarray] = None, ) -> None:
        background_color = background_color if background_color is not None else np.array([0.0, 134.0, 201.0])
        self.background_color = background_color.astype(np.float32) / 255
        options = "--background_color_red={} \
                    --background_color_green={} \
                    --background_color_blue={}".format(
            *self.background_color
        )

        self.connection_mode = p.GUI if render else p.DIRECT
        self.render_env = render
        if render:
            options += " --mp4=\"test.mp4\" --mp4fps=60"

        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        # self.dummy_collision_client = None
        # if dummy_client:
        #     self.dummy_collision_client = bc.BulletClient(connection_mode=p.DIRECT)
        # self.physics_client = bc.BulletClient(connection_mode=p.DIRECT, options=options)
        # self.dummy_collision_client = bc.BulletClient(connection_mode=p.GUI)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}
        self._string_idx = {}
        self._string_positions = {}

        if self.render_env:
            self.physics_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self.physics_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        self.physics_client.disconnect()
        # if self.dummy_collision_client is not None:
        #     self.dummy_collision_client.disconnect()

    def save_state(self) -> int:
        """Save the current simulation state.

        Returns:
            int: A state id assigned by PyBullet, which is the first non-negative
            integer available for indexing.
        """
        return self.physics_client.saveState()

    def restore_state(self, state_id: int) -> None:
        """Restore a simulation state.

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.restoreState(state_id)

    def remove_state(self, state_id: int) -> None:
        """Remove a simulation state. This will make this state_id available again for returning in save_state().

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.removeState(state_id)

    def remove_body(self, body_name: str) -> str:
        """Remove a body from the simulation.

        Args:
            body_name (str): Body unique name.
        Returns:
            str: The body name.
        """
        self.physics_client.removeBody(self._bodies_idx[body_name])
        del self._bodies_idx[body_name]

        return body_name

    def render(
            self,
            mode: str = "human",
            width: int = 720,
            height: int = 480,
            target_position: Optional[np.ndarray] = None,
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(3)
        if mode == "human":
            self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING)
            time.sleep(self.dt)  # wait to seems like real speed
        if mode == "rgb_array":
            if self.connection_mode == p.DIRECT:
                warnings.warn(
                    "The use of the render method is not recommended when the environment "
                    "has not been created with render=True. The rendering will probably be weird. "
                    "Prefer making the environment with option `render=True`. For example: "
                    "`env = gym.make('PandaReach-v3', render=True)`.",
                    UserWarning,
                )
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            return px

    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)

    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (x, y, z, w).
        """
        orientation = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)

    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.physics_client.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")

    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)

    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def get_joint_angles(self, body: str, joints: np.ndarray) -> np.ndarray:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """

        return np.array([i[0] for i in self.physics_client.getJointStates(self._bodies_idx[body], joints)])

    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]

    def get_joint_velocities(self, body: str, joints: np.ndarray) -> np.ndarray:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): Joint index in the body

        Returns:
            float: The velocity.
        """
        return np.array([i[1] for i in self.physics_client.getJointStates(self._bodies_idx[body], joints)])

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray,
                      physics_client=None) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if physics_client is None:
            physics_client = self.physics_client

        if len(orientation) == 3:
            orientation = physics_client.getQuaternionFromEuler(orientation)
        physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_base_velocity(self, body: str, velocity: np.ndarray, physics_client=None) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if physics_client is None:
            physics_client = self.physics_client

        physics_client.resetBaseVelocity(
            objectUniqueId=self._bodies_idx[body], linearVelocity=velocity
        )

    def set_base_pose_dummy(self, body_id: int, position: np.ndarray, orientation: np.ndarray,
                            physics_client=None) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if physics_client is None:
            physics_client = self.physics_client

        if len(orientation) == 3:
            orientation = physics_client.getQuaternionFromEuler(orientation)
        physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=body_id, posObj=position, ornObj=orientation
        )

    def set_base_velocity_dummy(self, body_id: int, velocity: np.ndarray, physics_client=None) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if physics_client is None:
            physics_client = self.physics_client

        physics_client.resetBaseVelocity(
            objectUniqueId=body_id, linearVelocity=velocity
        )

    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def control_joints(self, body: str, joints: np.ndarray, action: np.ndarray, forces: np.ndarray,
                       control_mode: int) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            action (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        match control_mode:
            case self.physics_client.POSITION_CONTROL:
                self.physics_client.setJointMotorControlArray(
                    self._bodies_idx[body],
                    jointIndices=joints,
                    controlMode=control_mode,
                    targetPositions=action,
                    forces=forces,
                )
            case self.physics_client.VELOCITY_CONTROL:
                self.physics_client.setJointMotorControlArray(
                    self._bodies_idx[body],
                    jointIndices=joints,
                    controlMode=control_mode,
                    targetVelocities=action,
                    forces=forces,
                )

    def inverse_kinematics(self, body: str, link: int, position: np.ndarray,
                           orientation: Optional[np.ndarray]) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        if orientation is not None:
            joint_state = self.physics_client.calculateInverseKinematics(
                bodyIndex=self._bodies_idx[body],
                endEffectorLinkIndex=link,
                targetPosition=position,
                targetOrientation=orientation,
            )
        else:
            joint_state = self.physics_client.calculateInverseKinematics(
                bodyIndex=self._bodies_idx[body],
                endEffectorLinkIndex=link,
                targetPosition=position,
                #maxNumIterations=1000,
                #residualThreshold=0.05
            )
        return np.array(joint_state)

    def place_visualizer(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> int:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        idx = self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)
        return idx

    def load_scenario(self, urdfs):
        indexes = []
        for body_name, kwargs in urdfs.items():
            indexes.append(self.loadURDF(body_name, **kwargs))

        return indexes

    def create_box(
            self,
            body_name: str,
            half_extents: np.ndarray,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = None,
            specular_color: Optional[np.ndarray] = None,
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            texture: Optional[str] = None,
            physics_client=None
    ) -> int:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        if physics_client is None:
            physics_client = self.physics_client

        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        idx = self._create_geometry(
            body_name,
            geom_type=physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
            physics_client=physics_client
        )
        if texture is not None:
            texture_path = os.path.join(panda_gym.assets.get_data_path(), texture)
            texture_uid = physics_client.loadTexture(texture_path)
            physics_client.changeVisualShape(self._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

        return idx

    def create_cylinder(
            self,
            body_name: str,
            radius: float,
            height: float,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = None,
            specular_color: Optional[np.ndarray] = None,
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            physics_client=None
    ) -> int:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        if physics_client is None:
            physics_client = self.physics_client

        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        idx = self._create_geometry(
            body_name,
            geom_type=physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

        return idx

    def create_sphere(
            self,
            body_name: str,
            radius: float,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = None,
            specular_color: Optional[np.ndarray] = None,
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            physics_client=None,
    ) -> int:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """

        if physics_client is None:
            physics_client = self.physics_client

        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = specular_color if specular_color is not None else np.zeros(3)
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        idx = self._create_geometry(
            body_name,
            geom_type=physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
            physics_client=physics_client
        )
        return idx

    def _create_geometry(
            self,
            body_name: str,
            geom_type: int,
            mass: float = 0.0,
            position: Optional[np.ndarray] = None,
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            visual_kwargs: Dict[str, Any] = {},
            collision_kwargs: Dict[str, Any] = {},
            physics_client=None
    ) -> int:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """

        if physics_client is None:
            physics_client = self.physics_client

        position = position if position is not None else np.zeros(3)
        baseVisualShapeIndex = physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        idx = physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if physics_client is self.physics_client:
            self._bodies_idx[body_name] = idx

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

        return idx

    def create_plane(self, z_offset: float, physics_client=None) -> int:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        if physics_client is None:
            physics_client = self.physics_client

        idx = self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
            physics_client=physics_client
        )

        return idx

    def create_table(
            self,
            length: float,
            width: float,
            height: float,
            x_offset: float = 0.0,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            physics_client=None
    ) -> int:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offet in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        if physics_client is None:
            physics_client = self.physics_client

        idx = self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.3, 0.3, 0.3, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            physics_client=physics_client
        )

        return idx

    def create_debug_text(
            self,
            text_name: str,
            text: str,
            position: np.ndarray = None,
            color: Optional[str] = np.array([0.0, 1.0, 0.0]),

    ) -> int:
        """Create a geometry.

        Args:
            text_name (str): The name of the floating text. Must be unique in the sim.
            text (str): Text to display.
            position (np.ndarray): The position, as (x, y, z). Defaults to a position on the upper left.
            color (str, optional): Color of the floating text.

        """

        idx = self._string_idx.get(text_name)
        if idx is None:
            position = np.array([0.5, 0.9, +1.0 - len(self._string_idx) / 20])
            idx = self._string_idx[text_name] = p.addUserDebugText(text=text,
                                                                   textSize=1, lifeTime=0,
                                                                   physicsClientId=self.physics_client._client,
                                                                   textPosition=position,
                                                                   textColorRGB=color)
            self._string_positions[text_name] = position

        else:
            position = self._string_positions[text_name]
            idx = self._string_idx[text_name] = p.addUserDebugText(text=text,
                                                                   textSize=1, lifeTime=0,
                                                                   physicsClientId=self.physics_client._client,
                                                                   textPosition=position,
                                                                   textColorRGB=color,
                                                                   replaceItemUniqueId=idx)

        return idx

    def create_debug_line(self, start, end, id=0, color=np.array([0, 1, 0])):
        p.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end, lineColorRGB=color, physicsClientId=id)

    def remove_debug_text(self, text_name):

        self.physics_client.removeUserDebugItem(itemUniqueId=self._string_idx[text_name],
                                                physicsClientId=self.physics_client._client)
        self._string_idx.pop(text_name)

    def remove_all_debug_text(self):
        self.physics_client.removeAllUserDebugItems(physicsClientId=self.physics_client._client)
        self._string_idx.clear()

    def set_debug_object_color(
            self,
            body_name: str,
            color: Optional[str] = np.array([0.0, 1.0, 0.0])):

        p.setDebugObjectColor(objectUniqueId=self._bodies_idx[body_name], objectDebugColorRGB=color,
                              physicsClientId=self.physics_client.client,
                              linkIndex=-1)

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )
