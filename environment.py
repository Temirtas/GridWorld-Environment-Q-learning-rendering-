from typing import Optional

import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.size = size
        self.render_mode = render_mode

        # Pygame state (created lazily on first render)
        self.window_size = 512  # size of the pygame window in pixels
        self.window = None
        self.clock = None

        # Positions are [x, y]; set in reset()
        self._agent_location = np.array([-1, -1], dtype=np.int64)
        self._target_location = np.array([-1, -1], dtype=np.int64)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        # x grows to the right, y grows upward
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": int(np.abs(self._agent_location - self._target_location).sum())}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=np.int64
            )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = False
        reward = 1 if terminated else 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render without specifying a render_mode. "
                "Use gym.make(..., render_mode='human'|'ansi'|'rgb_array')."
            )
            return None
        if self.render_mode == "ansi":
            return self._render_text()
        return self._render_frame()  # "rgb_array" returns the frame; "human" returns None

    def _render_text(self) -> str:
        rows = []
        for y in range(self.size - 1, -1, -1):  # top row first
            row = []
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row.append("A")
                elif np.array_equal([x, y], self._target_location):
                    row.append("T")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def _render_frame(self):
        import pygame  # imported lazily so the env works without pygame for other modes

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size  # pixels per grid square

        def to_px(loc):
            # y=0 is the bottom row, so flip y for screen coordinates
            return np.array([loc[0], self.size - 1 - loc[1]]) * pix

        # Target: red square
        pygame.draw.rect(
            canvas, (255, 0, 0), pygame.Rect(to_px(self._target_location), (pix, pix))
        )
        # Agent: blue circle
        pygame.draw.circle(
            canvas, (0, 0, 255), to_px(self._agent_location) + pix / 2, pix / 3
        )
        # Grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, pix * i), (self.window_size, pix * i), width=3)
            pygame.draw.line(canvas, (0, 0, 0), (pix * i, 0), (pix * i, self.window_size), width=3)

        if self.render_mode == "human":
            pygame.event.pump()  # keep the window responsive
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        # rgb_array: (H, W, 3) uint8
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,
)


if __name__ == "__main__":
    # Live pygame window with a random agent
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, render_mode="human")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        print(f"Episode {episode + 1} finished")
    env.close()