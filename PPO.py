#!/usr/bin/env python3

# This disgrace of a code was written using theft powered by ChatGPT
# TODO: Make your own

import numpy as np
import tensorflow as tf
import gym

from NMModel import NMModel


# Proximal Policy Optimization (PPO) Agent
class PPOAgent:
    def __init__(self, action_dim, actor, critic, lr_actor=1e-4,
                 lr_critic=1e-3, gamma=0.99, clip_epsilon=0.2, c1=1.0,
                 c2=0.01):
        self.actor = actor
        self.critic = critic
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)
        return np.squeeze(action, axis=0)

    def get_value(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return tf.squeeze(self.critic(state), axis=0)

    def compute_loss(self, states, actions, advantages, old_probs):
        with tf.GradientTape(persistent=True) as tape:
            # Actor loss
            new_probs = self.actor(states)
            entropy = -tf.reduce_sum(
                new_probs * tf.math.log(new_probs + 1e-10), axis=-1)
            ratio = new_probs / (old_probs + 1e-10)
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * advantages
            actor_loss = -tf.reduce_mean(
                tf.minimum(surrogate1, surrogate2) - self.c1 * entropy)

            # Critic loss
            value_pred = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(value_pred - advantages))

        # Compute gradients and update networks
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.optimizer_actor.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))

    def train(self, states, actions, advantages, old_probs):
        self.compute_loss(states, actions, advantages, old_probs)


# Main PPO training loop
def train_ppo(env_name, num_episodes, max_steps, actor, critic):
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = PPOAgent(action_dim, actor, critic)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        old_probs = []

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(agent.actor(
                tf.convert_to_tensor([state], dtype=tf.float32))[0])
            state = next_state

            if done:
                break

        discounted_rewards = []
        advantage = 0
        for r in rewards[::-1]:
            advantage = r + agent.gamma * advantage
            discounted_rewards.append(advantage)
        discounted_rewards.reverse()
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-8)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)

        agent.train(states, actions, discounted_rewards, old_probs)

        episode_reward = sum(rewards)
        print(f"Episode: {episode + 1}, Total Reward: {episode_reward}")


if __name__ == "__main__":
    env_name = 'Pendulum-v1'  # Change this to your desired environment
    num_episodes = 500
    max_steps = 200

    # Replace with your custom actor network
    custom_actor = NMModel()
    custom_critic = NMModel()

    train_ppo(
        env_name,
        num_episodes,
        max_steps,
        custom_actor,
        custom_critic
    )
