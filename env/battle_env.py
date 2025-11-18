import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BattleEnv(gym.Env):
    """
    탑다운 아레나 보스전 환경 (숫자 기반)
    - 에이전트: 보스
    - 플레이어: 스크립트/고정 패턴(지금은 단순함)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 300):
        super().__init__()

        # 아레나 크기 (픽셀 단위, 나중에 pygame에도 동일하게 사용 가능)
        self.WIDTH = 800
        self.HEIGHT = 600

        # 속도 / 쿨타임 기본값
        self.BASE_BOSS_SPEED = 5.0
        self.BASE_BULLET_SPEED = 7.0
        self.BASE_ATTACK_COOLDOWN = 20  # 스텝 단위

        # 에피소드 당 최대 스텝
        self.max_steps = max_steps

        # 관측 공간: [px, py, bx, by, dist, boss_speed, attack_cd_norm, player_hp_norm]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        # 행동 공간 (Discrete 5)
        # 0: 대기
        # 1: 플레이어 쪽으로 이동
        # 2: 플레이어와 거리 벌리기
        # 3: 좌우로 도는 움직임 (단순 회피 패턴)
        # 4: 공격 시도 (탄환 발사)
        self.action_space = spaces.Discrete(5)

        # 상태 변수들
        self.player_x = None
        self.player_y = None
        self.player_hp = None

        self.boss_x = None
        self.boss_y = None
        self.boss_speed = None
        self.attack_cooldown = None

        # 탄환 리스트: [(x, y, vx, vy), ...]
        self.bullets = []

        self.step_count = 0

    # ------------------------
    # 유틸 함수들
    # ------------------------
    def _reset_entities(self):
        # 플레이어: 아래쪽 중앙 근처
        self.player_x = self.WIDTH * 0.5
        self.player_y = self.HEIGHT * 0.8
        self.player_hp = 100.0

        # 보스: 위쪽 중앙
        self.boss_x = self.WIDTH * 0.5
        self.boss_y = self.HEIGHT * 0.2
        self.boss_speed = self.BASE_BOSS_SPEED

        self.attack_cooldown = 0
        self.bullets = []
        self.step_count = 0

    def _get_distance(self):
        dx = self.player_x - self.boss_x
        dy = self.player_y - self.boss_y
        return np.hypot(dx, dy)

    def _get_obs(self):
        dist = self._get_distance()
        max_dist = np.hypot(self.WIDTH, self.HEIGHT)

        obs = np.array([
            self.player_x / self.WIDTH,
            self.player_y / self.HEIGHT,
            self.boss_x / self.WIDTH,
            self.boss_y / self.HEIGHT,
            dist / max_dist,
            self.boss_speed / 10.0,              # 대충 0~1 스케일
            self.attack_cooldown / self.BASE_ATTACK_COOLDOWN if self.BASE_ATTACK_COOLDOWN > 0 else 0.0,
            self.player_hp / 100.0
        ], dtype=np.float32)

        return obs

    def _move_boss_towards_player(self):
        dx = self.player_x - self.boss_x
        dy = self.player_y - self.boss_y
        dist = np.hypot(dx, dy)
        if dist > 1e-5:
            vx = self.boss_speed * dx / dist
            vy = self.boss_speed * dy / dist
            self.boss_x += vx
            self.boss_y += vy

    def _move_boss_away_from_player(self):
        dx = self.boss_x - self.player_x
        dy = self.boss_y - self.player_y
        dist = np.hypot(dx, dy)
        if dist > 1e-5:
            vx = self.boss_speed * dx / dist
            vy = self.boss_speed * dy / dist
            self.boss_x += vx
            self.boss_y += vy

    def _move_boss_strafe(self):
        # 단순 좌우 움직임 (y는 거의 유지)
        self.boss_x += self.boss_speed * np.sign(
            np.sin(self.step_count / 20.0)
        )

    def _clamp_positions(self):
        self.player_x = np.clip(self.player_x, 0, self.WIDTH)
        self.player_y = np.clip(self.player_y, 0, self.HEIGHT)
        self.boss_x = np.clip(self.boss_x, 0, self.WIDTH)
        self.boss_y = np.clip(self.boss_y, 0, self.HEIGHT)

    def _spawn_bullet_towards_player(self):
        dx = self.player_x - self.boss_x
        dy = self.player_y - self.boss_y
        dist = np.hypot(dx, dy)
        if dist < 1e-5:
            return

        vx = self.BASE_BULLET_SPEED * dx / dist
        vy = self.BASE_BULLET_SPEED * dy / dist
        self.bullets.append([self.boss_x, self.boss_y, vx, vy])

    def _update_bullets(self):
        hit = False
        new_bullets = []
        for (x, y, vx, vy) in self.bullets:
            x += vx
            y += vy

            # 화면 밖으로 나가면 삭제
            if x < 0 or x > self.WIDTH or y < 0 or y > self.HEIGHT:
                continue

            # 플레이어와 충돌 체크 (단순 반지름 거리 기준)
            dx = x - self.player_x
            dy = y - self.player_y
            dist = np.hypot(dx, dy)
            if dist < 30:  # 충돌 반경
                hit = True
                self.player_hp -= 10.0
                continue  # 이 탄환은 사라짐

            new_bullets.append([x, y, vx, vy])

        self.bullets = new_bullets
        return hit

    # ------------------------
    # Gym 인터페이스 구현부
    # ------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_entities()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # 보스 행동
        if action == 0:
            # 대기
            pass
        elif action == 1:
            # 플레이어 쪽으로 이동
            self._move_boss_towards_player()
        elif action == 2:
            # 플레이어와 거리 벌리기
            self._move_boss_away_from_player()
        elif action == 3:
            # 좌우로 도는 움직임
            self._move_boss_strafe()
        elif action == 4:
            # 공격 시도 (쿨타임 체크)
            if self.attack_cooldown <= 0:
                self._spawn_bullet_towards_player()
                self.attack_cooldown = self.BASE_ATTACK_COOLDOWN

        # 쿨타임 감소
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # 플레이어는 일단 고정 혹은 아주 약한 랜덤 이동으로 처리 (추후 사람 조작으로 교체)
        # 여기서는 간단히 좌우로 조금씩 움직이는 패턴
        self.player_x += 2.0 * np.sign(np.sin(self.step_count / 15.0))

        # 위치 범위 제한
        self._clamp_positions()

        # 탄환 업데이트 및 명중 여부 체크
        hit = self._update_bullets()

        # 보상 계산
        reward = 0.0
        if hit:
            reward += 1.0
        # 시간 지날수록 약간의 패널티 (빨리 끝낼수록 유리)
        reward -= 0.01

        # 종료 조건
        terminated = False
        truncated = False

        if self.player_hp <= 0:
            terminated = True
            reward += 10.0  # 플레이어 쓰러뜨리면 큰 보상
        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 5.0  # 시간 끌다가 못 잡으면 패널티

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # 간단 테스트용
    env = BattleEnv()
    obs, info = env.reset()
    print("초기 관측:", obs)

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={t}, action={action}, reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            break
