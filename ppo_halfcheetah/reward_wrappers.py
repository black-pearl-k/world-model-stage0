"""HalfCheetah-v5 的步态奖励包装器。

这个文件不修改 Gymnasium 或 Stable-Baselines3 的源码，而是用一个
``gym.Wrapper`` 在原始环境外面加一层奖励。

原始 HalfCheetah 奖励只关心：
    前进速度 - 动作代价

所以模型可以学会“翻过来用头/背滑行”，仍然拿到很高的分数。

这里额外加入：
    1. 身体水平奖励：cos(rooty)
    2. 躯干/头碰地惩罚
    3. 高度偏离惩罚
    4. 可选的目标速度上限（第二阶段再用）

注意：
    - 训练时使用这个 wrapper；
    - 最终评测时仍然用没有 wrapper 的环境，得到原始 HalfCheetah 回报；
    - 训练 reward 和原始 reward 不是同一个尺度，不能直接和 baseline 比较。
"""

from __future__ import annotations  # 让类型注解延迟求值

import math  # 计算 cos
from typing import Any  # 类型注解

import gymnasium as gym  # 环境基类
import mujoco  # 读取 MuJoCo 的 geom 和 contact 信息
import numpy as np  # 处理动作向量


class GaitRewardWrapper(gym.Wrapper):
    """在 HalfCheetah-v5 上加入姿态和接触惩罚。

    参数
    ----
    env:
        原始 HalfCheetah-v5 环境。
    w_upright:
        身体水平奖励权重。rooty=0 时 cos=1，翻过来时 cos≈-1。
    w_body_contact:
        躯干或头碰地时，每步扣多少分。
    w_height:
        身体高度偏离目标高度时，每步扣多少分。
    height_tolerance:
        高度允许的自然波动范围，超过这个范围才扣分。
    target_speed:
        目标速度。None 表示不修改原始前进奖励（第一阶段建议用 None）。
        设置成 1.5 后，超过 1.5 m/s 的部分会被压制。
    w_speed_over:
        超过目标速度时，额外惩罚的权重。
        1.0 表示超过目标后前进奖励基本封顶；2.0 表示超速会被明显扣分。
    target_z:
        固定目标高度。None 表示每次 reset 时记录初始高度作为目标。
    """

    def __init__(
        self,
        env: gym.Env,
        w_upright: float = 0.5,
        w_body_contact: float = 1.0,
        w_height: float = 0.3,
        height_tolerance: float = 0.15,
        target_speed: float | None = None,
        w_speed_over: float = 1.0,
        target_z: float | None = None,
    ) -> None:
        super().__init__(env)  # 保存原始环境

        self.w_upright = float(w_upright)  # 身体水平奖励权重
        self.w_body_contact = float(w_body_contact)  # 躯干/头碰地惩罚权重
        self.w_height = float(w_height)  # 高度偏离惩罚权重
        self.height_tolerance = float(height_tolerance)  # 高度允许波动范围
        self.target_speed = target_speed  # 目标速度；None 表示不启用
        self.w_speed_over = float(w_speed_over)  # 超速惩罚权重
        self._fixed_target_z = target_z  # 用户指定的固定目标高度
        self.target_z = target_z  # 当前目标高度；reset 时更新

        model = self.env.unwrapped.model  # 取出底层 MuJoCo 模型
        self.floor_id = self._geom_id(model, "floor")  # 地面 geom
        self.torso_id = self._geom_id(model, "torso")  # 躯干 geom
        self.head_id = self._geom_id(model, "head")  # 头部 geom

    @staticmethod
    def _geom_id(model: mujoco.MjModel, name: str) -> int:
        """按名字查找 geom 编号，找不到时给出清楚的错误。"""
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {name}")
        return int(geom_id)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """重置环境，并记录这一局的目标高度。"""
        obs, info = self.env.reset(**kwargs)

        if self._fixed_target_z is None:
            # 默认拿 reset 后的初始高度作为目标，避免把“正常跑动时的高度”
            # 硬编码成一个不一定适合当前模型的值。
            self.target_z = float(self.env.unwrapped.data.qpos[1])
        else:
            self.target_z = float(self._fixed_target_z)

        return obs, info

    def _body_contact(self) -> float:
        """检查躯干或头是否碰到地面；脚碰地是正常的，不罚。"""
        data = self.env.unwrapped.data  # 当前 MuJoCo 数据

        for i in range(data.ncon):  # 遍历当前所有接触点
            contact = data.contact[i]  # 第 i 个接触
            geom_pair = {int(contact.geom1), int(contact.geom2)}  # 两个 geom 的编号

            if self.floor_id in geom_pair and (
                self.torso_id in geom_pair or self.head_id in geom_pair
            ):
                return 1.0  # 躯干或头碰地，返回 1

        return 0.0  # 没有不允许的接触

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """执行一步，并返回塑形后的奖励。"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data  # 当前状态
        rootz = float(data.qpos[1])  # 躯干高度
        rooty = float(data.qpos[2])  # 躯干俯仰角
        x_velocity = float(info.get("x_velocity", 0.0))  # 前进速度

        # 正常流程一定先 reset；这里只是防止误用 step 时 target_z 还是 None。
        if self.target_z is None:
            self.target_z = rootz

        upright = math.cos(rooty)  # 水平时接近 1，翻过来时接近 -1
        body_contact = self._body_contact()  # 躯干/头是否碰地
        height_error = max(
            0.0,
            abs(rootz - self.target_z) - self.height_tolerance,
        )  # 只有超过允许范围才罚高度

        # 第一阶段 target_speed=None，base_reward 就是原始奖励。
        # 第二阶段打开 target_speed 后，会替换原始的前进奖励，避免
        # “一边奖励速度，一边又惩罚速度”互相打架。
        if self.target_speed is None:
            base_reward = float(reward)
        else:
            reward_ctrl = float(info.get("reward_ctrl", 0.0))
            if x_velocity <= self.target_speed:
                forward_reward = x_velocity
            else:
                forward_reward = self.target_speed - self.w_speed_over * (
                    x_velocity - self.target_speed
                )
            base_reward = forward_reward + reward_ctrl

        shaped_reward = (
            base_reward
            + self.w_upright * upright
            - self.w_body_contact * body_contact
            - self.w_height * height_error
        )

        # 把这些量写进 info，训练日志和视频统计都可以直接读取。
        info["reward_original"] = float(reward)
        info["reward_base"] = float(base_reward)
        info["reward_shaped"] = float(shaped_reward)
        info["upright"] = float(upright)
        info["body_contact"] = float(body_contact)
        info["height_error"] = float(height_error)
        info["rootz"] = float(rootz)
        info["rooty"] = float(rooty)
        info["x_velocity"] = float(x_velocity)
        info["target_z"] = float(self.target_z)

        return obs, float(shaped_reward), terminated, truncated, info


class StandRewardWrapper(GaitRewardWrapper):
    """第一阶段专用：只练习“站稳”，完全不奖励前进。

    和 ``GaitRewardWrapper`` 的区别：

    - 不再使用原始 HalfCheetah 奖励；
    - 不奖励前进速度；
    - 反而惩罚 x 方向速度，让它不要跑；
    - 额外惩罚俯仰角速度、根节点速度、关节速度、动作变化；
    - 头部接触和躯干接触分开惩罚，头部更重。

    目标不是拿高分，而是先让模型学会：

        rooty 接近 0
        身体大致水平
        躯干和头不碰地
        不要翻滚
        不要猛蹬
        不要往前冲

    注意：HalfCheetah 的“站起来”不是人形直立，而是身体大致水平、
    腿在身体下面支撑。这个 wrapper 的目标高度来自 reset 时的初始高度。
    """

    def __init__(
        self,
        env: gym.Env,
        w_upright: float = 1.0,
        w_torso_contact: float = 2.0,
        w_head_contact: float = 5.0,
        w_height: float = 0.5,
        height_tolerance: float = 0.15,
        w_x_velocity: float = 0.5,
        w_root_velocity: float = 0.2,
        w_joint_velocity: float = 0.02,
        w_action: float = 0.05,
        w_action_rate: float = 0.05,
    ) -> None:
        # 复用父类的 geom 查找、target_z 记录和 reset 逻辑。
        super().__init__(
            env,
            w_upright=w_upright,
            w_body_contact=0.0,
            w_height=w_height,
            height_tolerance=height_tolerance,
            target_speed=None,
        )
        self.w_torso_contact = float(w_torso_contact)
        self.w_head_contact = float(w_head_contact)
        self.w_x_velocity = float(w_x_velocity)
        self.w_root_velocity = float(w_root_velocity)
        self.w_joint_velocity = float(w_joint_velocity)
        self.w_action = float(w_action)
        self.w_action_rate = float(w_action_rate)
        self.prev_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """重置环境，同时清空动作历史。"""
        obs, info = super().reset(**kwargs)
        self.prev_action = None
        return obs, info

    def _contact_flags(self) -> tuple[float, float]:
        """分别检查躯干碰地和头碰地。"""
        data = self.env.unwrapped.data
        torso_contact = 0.0
        head_contact = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]
            geom_pair = {int(contact.geom1), int(contact.geom2)}

            if self.floor_id not in geom_pair:
                continue
            if self.torso_id in geom_pair:
                torso_contact = 1.0
            if self.head_id in geom_pair:
                head_contact = 1.0

        return torso_contact, head_contact

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """只返回站立奖励，不使用原始前进奖励。"""
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        rootz = float(data.qpos[1])
        rooty = float(data.qpos[2])
        x_velocity = float(info.get("x_velocity", 0.0))
        qvel = np.asarray(data.qvel, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)

        if self.target_z is None:
            self.target_z = rootz

        upright = math.cos(rooty)
        torso_contact, head_contact = self._contact_flags()
        body_contact = max(torso_contact, head_contact)
        height_error = max(
            0.0,
            abs(rootz - self.target_z) - self.height_tolerance,
        )

        root_velocity = float(np.sum(qvel[:3] ** 2))
        joint_velocity = float(np.sum(qvel[3:] ** 2))
        action_cost = float(np.sum(action_array**2))

        if self.prev_action is None:
            action_rate = 0.0
        else:
            action_rate = float(np.sum((action_array - self.prev_action) ** 2))

        # base_reward 固定为 0：原始前进奖励被彻底去掉。
        # 站直、少碰地、少动、动作平滑才得分。
        stand_reward = (
            self.w_upright * upright
            - self.w_torso_contact * torso_contact
            - self.w_head_contact * head_contact
            - self.w_height * height_error
            - self.w_x_velocity * x_velocity**2
            - self.w_root_velocity * root_velocity
            - self.w_joint_velocity * joint_velocity
            - self.w_action * action_cost
            - self.w_action_rate * action_rate
        )

        self.prev_action = action_array.copy()

        info["reward_original"] = float(original_reward)
        info["reward_base"] = 0.0
        info["reward_shaped"] = float(stand_reward)
        info["upright"] = float(upright)
        info["torso_contact"] = float(torso_contact)
        info["head_contact"] = float(head_contact)
        info["body_contact"] = float(body_contact)
        info["height_error"] = float(height_error)
        info["rootz"] = float(rootz)
        info["rooty"] = float(rooty)
        info["x_velocity"] = float(x_velocity)
        info["root_velocity"] = float(root_velocity)
        info["joint_velocity"] = float(joint_velocity)
        info["action_cost"] = float(action_cost)
        info["action_rate"] = float(action_rate)
        info["target_z"] = float(self.target_z)

        return obs, float(stand_reward), terminated, truncated, info
