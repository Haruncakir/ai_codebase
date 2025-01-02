import pygame
import torch
from stable_baselines3.common.callbacks import BaseCallback
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns


class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=10, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render()
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        return True

class XAIPongWrapper:
    def __init__(self, model_):
        self.model = model_
        self.feature_names = ['Ball X', 'Ball Y', 'Ball DX', 'Ball DY', 'Paddle Y', 'Opponent Y']
        self.ig = IntegratedGradients(self.model.policy.predict)

    def get_attribution(self, observation):
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Calculate attributions using Integrated Gradients
        attributions = self.ig.attribute(obs_tensor)

        return attributions.detach().numpy()[0]

    def visualize_attribution(self, observation, save_path=None):
        attributions = self.get_attribution(observation)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.feature_names, y=attributions)
        plt.title('Feature Importance in AI Decision Making')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

        return attributions


def train_pong_ai(env, total_timesteps=100000):
    # Create the model
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                device='cpu')

    # Create the callback
    render_callback = RenderCallback(env, render_freq=10)

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=render_callback)
    return model


def evaluate_and_explain(_env, _model, _xai_wrapper, n_episodes=5):
    for episode in range(n_episodes):
        obs, _ = _env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Get model prediction and explanation
            action, _ = _model.predict(obs)

            # Generate and visualize XAI explanation every 10 steps
            if step % 10 == 0:
                attributions = _xai_wrapper.visualize_attribution(
                    obs, f"explanation_ep{episode}_step{step}.png"
                )
                print(f"\nStep {step} Feature Importance:")
                for name, attr in zip(_xai_wrapper.feature_names, attributions):
                    print(f"{name}: {attr:.4f}")

            # Take action in environment
            obs, reward, done, truncated, info = _env.step(action)
            episode_reward += reward
            step += 1

            _env.render()

        print(f"Episode {episode + 1} reward: {episode_reward}")


if __name__ == "__main__":
    from pong_env import PongEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3 import PPO

    # Create and wrap the environment
    env = PongEnv()
    env = Monitor(env)

    # Train the model
    model = train_pong_ai(env)

    # Optional: Save the trained model
    model.save("pong_model")

    # Close the environment
    env.close()