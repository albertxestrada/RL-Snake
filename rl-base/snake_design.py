"""Diseño del problema del Snake: observación y recompensa  (parte de la entrega).

Aquí decidís QUÉ ve el agente y QUÉ se le premia. El mismo diseño lo usan vuestro
REINFORCE (lab_3_3_reinforce.py) y las referencias de SB3 (lab_3_3_referencia_sb3.py),
así que la comparación entre algoritmos es justa.

    TODO 6  SnakeObservation   una observación mejor que la "original"
    TODO 7  SnakeReward        una recompensa propia (opcional pero valorada)

Uso desde los scripts:   --obs original|custom   --reward sparse|distance|custom

Para medir el rendimiento NO se usa el retorno (depende de la recompensa que elijáis),
sino el número de manzanas por partida (info["score"]): así se pueden comparar diseños.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import snake_env  # noqa: F401  (registra DeustoSnake-v0)
from snake_env import ENV_ID, MOVES


# =========================================================================== TODO 6
class SnakeObservation(gym.ObservationWrapper):
    """Observación diseñada por vosotros.

    Tenéis acceso al estado completo en self.env.unwrapped:
        snake (lista de (x, y), cabeza primero) · apple (x, y) · direction (0–3)
        grid_size · steps_since_apple
    y a la tabla MOVES (acción -> desplazamiento).

    Preguntas para guiar el diseño (justificadlas en el informe):
      - ¿Necesita el agente coordenadas absolutas o le basta información relativa a la cabeza?
      - ¿Cómo sabe si el siguiente paso en cada dirección es mortal?
      - ¿Qué tamaño de observación es razonable para una red pequeña?
    """

    # OBS_SIZE = Es un 15 porque es la suma de las piezas que elegimos darle a la serpiente:
    #
    #   piezas                                  cuántos   por qué ese número
    #   --------------------------------------  --------  ------------------------------------
    #   peligro si voy a izq/der/abajo/arriba      4      hay 4 acciones: 1 pregunta por acción
    #   dirección actual (one-hot)                 4      una de 4 direcciones: 4 casillas, solo una vale 1
    #   manzana está a izq/der/abajo/arriba        4      4 preguntas de sí/no: una por dirección
    #   vector cabeza -> manzana (Δx, Δy)          2      un valor por eje (x, y), normalizado
    #   longitud de la serpiente                   1      un único valor, normalizado por el tamaño máximo posible
    OBS_SIZE = 15

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.OBS_SIZE,), dtype=np.float32)

    @staticmethod
    def _is_deadly(u, cell: tuple[int, int]) -> bool:
        """True si meter la cabeza en `cell` termina la partida."""
        x, y = cell
        if not (0 <= x < u.grid_size and 0 <= y < u.grid_size):
            return True                                   # Si se salio del grid, muerto
        return cell in u.snake[:-1]                       # Si toca una casilla asociada a la serpiente, muerto

    def observation(self, obs: np.ndarray) -> np.ndarray:
        u = self.env.unwrapped
        n = u.grid_size
        hx, hy = u.snake[0]
        ax, ay = u.apple

        # Revisa si a la der, izq, arr, aba hay peligros, para reportarlo en el vector de observación
        danger = [float(self._is_deadly(u, (hx + dx, hy + dy)))
                  for dx, dy in (MOVES[a] for a in range(4))]

        # Rellena el one hot de la dirección actual de la serpiente, segun la acción que tomó el agente.
        direction = [1.0 if u.direction == a else 0.0 for a in range(4)]

        # revisa si la manzana esta a la izquierda, derecha, arriba o abajo de la cabeza
        apple_side = [float(ax < hx), float(ax > hx), float(ay > hy), float(ay < hy)]

        # Distancia entre la cabeza y la manzana, normalizada por el tamaño del grid 
        # (para mantener el observation space entre -1 y 1)
        rel = [(ax - hx) / (n - 1), (ay - hy) / (n - 1)]

        # Longitud de la serpiente, normalizada por el tamaño máximo posible (n * n)
        length = [len(u.snake) / (n * n)]

        return np.clip(np.array(danger + direction + apple_side + rel + length, dtype=np.float32), -1.0, 1.0)


# =========================================================================== TODO 7
class SnakeReward(gym.Wrapper):
    """Recompensa diseñada por vosotros (se aplica sobre la recompensa "sparse").

    Recordad la U2 (diseño de recompensas, efecto cobra): el agente maximizará
    exactamente lo que premiéis, aunque no sea lo que queríais.
    """

    # Incentivo si se acerca o se aleja (negativo) de la manzana
    SHAPING = 0.05
    # Coste por paso (-), para evitar que el agente alargue la partida innecesariamente
    STEP_COST = 0.002

    @staticmethod
    def _apple_distance(u) -> int:
        hx, hy = u.snake[0]
        # Calcula la distancia Manhattan entre la cabeza y la manzana,
        # usando la fórmula |x1 - x2| + |y1 - y2|. (Distancia Manhattan)
        return abs(u.apple[0] - hx) + abs(u.apple[1] - hy)

    def step(self, action):
        u = self.env.unwrapped
        old_dist = self._apple_distance(u)                  # antes de mover
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and reward == 0.0:                # paso común: ni comió ni murió
            new_dist = self._apple_distance(u)
            reward += self.SHAPING * (old_dist - new_dist)  # +SHAPING si se acerca, -SHAPING si se aleja
            reward -= self.STEP_COST # Aplica el coste por un paso más
        return obs, reward, terminated, truncated, info


# =========================================================================== (hecho)
def make_snake_env(obs: str = "original", reward: str = "sparse", grid_size: int = 12,
                   render_mode: str | None = None) -> gym.Env:
    """Crea el Snake con el diseño elegido."""
    base_reward = reward if reward in ("sparse", "distance") else "sparse"
    env = gym.make(ENV_ID, grid_size=grid_size, reward_mode=base_reward, render_mode=render_mode)
    if obs == "custom":
        env = SnakeObservation(env)
    elif obs != "original":
        raise ValueError("obs debe ser 'original' o 'custom'")
    if reward == "custom":
        env = SnakeReward(env)
    elif reward not in ("sparse", "distance"):
        raise ValueError("reward debe ser 'sparse', 'distance' o 'custom'")
    return env


def evaluate_snake(act, env: gym.Env, n_episodes: int, seed: int) -> tuple[list[float], list[int]]:
    """Juega n partidas con act(obs) -> acción. Devuelve (retornos, manzanas por partida)."""
    returns, scores = [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += float(r)
            done = term or trunc
        returns.append(ret)
        scores.append(int(info["score"]))
    return returns, scores


def snake_eval_callback(eval_env: gym.Env, out_dir, eval_every: int, n_episodes: int, seed: int):
    """Callback de SB3 que evalúa en manzanas por partida y guarda evaluations.npz
    con el mismo formato que lab_3_3_reinforce.py (results = retornos, scores = manzanas)."""
    from stable_baselines3.common.callbacks import BaseCallback

    from common3 import save_evaluations

    class SnakeEval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.t, self.r, self.s = [], [], []

        def _evaluate(self):
            act = lambda o: int(self.model.predict(o, deterministic=True)[0])
            rets, scores = evaluate_snake(act, eval_env, n_episodes, seed)
            self.t.append(self.num_timesteps); self.r.append(rets); self.s.append(scores)
            save_evaluations(out_dir, self.t, self.r, scores=self.s)
            self.logger.record("eval/mean_score", float(np.mean(scores)))
            self.logger.record("eval/mean_reward", float(np.mean(rets)))

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self._evaluate()
                self.next_eval += eval_every
            return True

        def _on_training_end(self) -> None:
            self._evaluate()

    return SnakeEval()


def check_design(obs: str, reward: str, grid_size: int = 12) -> None:
    """Comprobación rápida de un diseño: formas, límites y una partida aleatoria."""
    env = make_snake_env(obs, reward, grid_size)
    o, _ = env.reset(seed=0)
    assert isinstance(env.observation_space, spaces.Box), "la observación debe ser un spaces.Box"
    assert o.dtype == np.float32, f"la observación debe ser float32, no {o.dtype}"
    assert o.shape == env.observation_space.shape, f"forma {o.shape} ≠ observation_space {env.observation_space.shape}"
    done, steps = False, 0
    while not done:
        o, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(o), f"observación fuera de observation_space: {o}"
        assert np.isfinite(r), "recompensa no finita"
        done, steps = term or trunc, steps + 1
    print(f"  [OK] diseño obs={obs}, reward={reward}: observación de tamaño {o.shape[0]}, partida de {steps} pasos")
