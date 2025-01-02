import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv, self).__init__()

        # Initialize Pygame
        pygame.init()

        # Constants
        self.WIDTH = 800
        self.HEIGHT = 600
        self.PADDLE_WIDTH = 15
        self.PADDLE_HEIGHT = 90
        self.BALL_SIZE = 15
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.PADDLE_SPEED = 7
        self.BALL_SPEED = 5

        # Pygame setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong with AI")
        self.clock = pygame.time.Clock()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.BALL_SPEED, -self.BALL_SPEED, 0, 0]),
            high=np.array([self.WIDTH, self.HEIGHT, self.BALL_SPEED, self.BALL_SPEED,
                           self.HEIGHT, self.HEIGHT]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset paddle positions
        self.paddle_y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.opponent_y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2

        # Reset ball
        self._reset_ball()

        # Reset scores
        self.player_score = 0
        self.opponent_score = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """Return the current observation of the environment."""
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_dx,
            self.ball_dy,
            self.paddle_y,
            self.opponent_y
        ], dtype=np.float32)

    def step(self, action):
        """Execute one time step within the environment."""
        reward = 0
        terminated = False
        truncated = False

        # Handle opponent paddle movement
        self._move_opponent_paddle(action)

        # Move player paddle based on ball position
        self._move_player_paddle()

        # Update ball position
        self._update_ball_position()

        # Check for collisions and scoring
        reward, terminated = self._check_collisions_and_score(reward)

        return self._get_obs(), reward, terminated, truncated, {}

    def _move_opponent_paddle(self, action):
        """Move the opponent paddle based on the action taken."""
        if action == 1 and self.opponent_y > 0:  # Move up
            self.opponent_y -= self.PADDLE_SPEED
        elif action == 2 and self.opponent_y < self.HEIGHT - self.PADDLE_HEIGHT:  # Move down
            self.opponent_y += self.PADDLE_SPEED

    def _move_player_paddle(self):
        """Move the player paddle based on the ball's position."""
        if self.ball_y > self.paddle_y + self.PADDLE_HEIGHT:
            self.paddle_y += self.PADDLE_SPEED
        elif self.ball_y < self.paddle_y:
            self.paddle_y -= self.PADDLE_SPEED

        # Clamp paddle position
        self.paddle_y = np.clip(self.paddle_y, 0, self.HEIGHT - self.PADDLE_HEIGHT)

    def _update_ball_position(self):
        """Update the ball's position based on its velocity."""
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collision with top and bottom
        if self.ball_y <= 0 or self.ball_y >= self.HEIGHT - self.BALL_SIZE:
            self.ball_dy *= -1

    def _check_collisions_and_score(self, reward):
        """Check for collisions and update scores accordingly."""
        player_paddle = pygame.Rect(50, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        opponent_paddle = pygame.Rect(self.WIDTH - 50 - self.PADDLE_WIDTH,
                                      self.opponent_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        if ball_rect.colliderect(player_paddle) or ball_rect.colliderect(opponent_paddle):
            self.ball_dx *= -1
            reward += 1  # Reward for hitting the ball

        # Ball out of bounds
        if self.ball_x <= 0:
            self.opponent_score += 1
            reward -= 5  # Penalty for losing point
            terminated = self.opponent_score >= 5
            self._reset_ball()
        elif self.ball_x >= self.WIDTH - self.BALL_SIZE:
            self.player_score += 1
            reward += 5  # Reward for scoring
            terminated = self.player_score >= 5
            self._reset_ball()
        else:
            terminated = False

        return reward, terminated

    def _reset_ball(self):
        """Reset the ball to the center of the screen with a random direction."""
        self.ball_x = self.WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.HEIGHT // 2 - self.BALL_SIZE // 2
        self.ball_dx = self.BALL_SPEED * np.random.choice([1, -1])
        self.ball_dy = self.BALL_SPEED * np.random.choice([1, -1])

    def render(self):
        """Render the current state of the environment."""
        self.screen.fill(self.BLACK)

        # Draw paddles
        pygame.draw.rect(self.screen, self.WHITE,
                         pygame.Rect(50, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, self.WHITE,
                         pygame.Rect(self.WIDTH - 50 - self.PADDLE_WIDTH,
                                     self.opponent_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))

        # Draw ball
        pygame.draw.rect(self.screen, self.WHITE,
                         pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        # Draw scores
        font = pygame.font.Font(None, 36)
        player_text = font.render(str(self.player_score), True, self.WHITE)
        opponent_text = font.render(str(self.opponent_score), True, self.WHITE)
        self.screen.blit(player_text, (self.WIDTH // 4, 20))
        self.screen.blit(opponent_text, (3 * self.WIDTH // 4, 20))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """Close the Pygame window and clean up resources."""
        pygame.quit()