# REINFORCE en el Snake · Unidad 3 · Aprendizaje por Refuerzo

Implementación de REINFORCE desde cero (PyTorch + Gymnasium) aplicada a un Snake de 12 × 12, con observación y recompensa diseñadas por el equipo. Se compara con A2C y PPO de Stable-Baselines3.
Máster en Computación y Sistemas Inteligentes · Universidad de Deusto.

**Equipo:** [Nombre 1] · [Nombre 2] · [Nombre 3] · [Nombre 4]

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python snake_env.py                                   # comprobación rápida del entorno
```

Python 3.10–3.12. Todos los comandos se ejecutan desde la raíz del repo. Los resultados se guardan en `logs/`.

## Estructura

| Archivo | Función |
|---|---|
| `lab_3_3_reinforce.py` | REINFORCE: red de política, reward-to-go, episodios, pérdida y baseline |
| `snake_design.py` | Observación (`SnakeObservation`) y recompensa (`SnakeReward`) propias |
| `snake_env.py` | Entorno `DeustoSnake-v0` |
| `lab_3_3_referencia_sb3.py` | A2C y PPO de SB3 con el mismo diseño y presupuesto |
| `common3.py` | Semillas, rutas y guardado de resultados |
| `plot_results.py` | Curvas medias, IC 95 % por bootstrap e IQM |

## Uso

Autocomprobaciones de cada TODO:

```bash
python lab_3_3_reinforce.py --check
```

Prueba en CartPole (para descartar fallos del algoritmo):

```bash
python lab_3_3_reinforce.py --env CartPole-v1 --seeds 0 --steps 50000
```

Experimentos (5 semillas, 200k pasos cada uno):

```bash
# REINFORCE
python lab_3_3_reinforce.py --seeds 0-4 --obs original --reward sparse
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward sparse
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward sparse --baseline
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward custom --baseline

# Referencias de SB3
python lab_3_3_referencia_sb3.py --seeds 0-4 --obs custom --reward sparse
```

Ablaciones:

```bash
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward sparse --no-normalize
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward sparse --lr 3e-4
python lab_3_3_reinforce.py --seeds 0-4 --obs custom --reward distance --baseline
```

Gráficas y tabla (la métrica es **manzanas por partida**, no el retorno):

```bash
python plot_results.py logs/lab_3_3 --metric score
```

Ver jugar a un agente entrenado:

```bash
python lab_3_3_reinforce.py --obs custom --watch logs/lab_3_3/reinforce_baseline_custom_sparse/seed_0/policy.pt
python lab_3_3_reinforce.py --obs custom --watch <policy.pt> --gif partida.gif     # GIF en lugar de ventana
```

TensorBoard: `tensorboard --logdir logs/tb`

## Estructura de resultados

```
logs/lab_3_3/<configuración>/seed_<k>/evaluations.npz   curva de evaluación
logs/lab_3_3/<configuración>/seed_<k>/policy.pt         política entrenada
logs/lab_3_3/<configuración>/seed_<k>/config.json       hiperparámetros usados
logs/tb/lab_3_3/                                        TensorBoard
```

## Diseño del Snake

**Observación (`custom`):** [describir: tamaño, qué contiene, por qué]

**Recompensa (`custom`):** [describir: términos, valores y justificación]

## Resultados

![Comparación](logs/lab_3_3.png)

| Configuración | Semillas | Manzanas (media) | IC 95 % | IQM |
|---|---|---|---|---|
| reinforce_original_sparse | 5 | [ ] | [ ] | [ ] |
| reinforce_custom_sparse | 5 | [ ] | [ ] | [ ] |
| reinforce_baseline_custom_sparse | 5 | [ ] | [ ] | [ ] |
| reinforce_baseline_custom_custom | 5 | [ ] | [ ] | [ ] |
| ppo_custom_sparse | 5 | [ ] | [ ] | [ ] |
| a2c_custom_sparse | 5 | [ ] | [ ] | [ ] |

Media de las últimas 3 evaluaciones de cada semilla.

## Contribuciones

| Integrante | Código | Experimentos | Informe |
|---|---|---|---|
| [Nombre 1] | `PolicyNet`, `collect_episode` | [ ] | [ ] |
| [Nombre 2] | `reward_to_go`, `policy_loss` | [ ] | [ ] |
| [Nombre 3] | `ValueNet`, baseline | [ ] | [ ] |
| [Nombre 4] | `SnakeObservation`, `SnakeReward` | [ ] | [ ] |

## Reproducibilidad

Semillas 0–4, 200 000 pasos de entorno, 20 partidas de evaluación cada 25 000 pasos (política determinista). Hiperparámetros de cada ejecución en su `config.json`.