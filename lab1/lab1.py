import os
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

map_name = "citymap.txt"
result_dir = "results"
seed_list = [11, 22, 33, 44, 55]
spawn_chances = [0.005, 0.01, 0.02, 0.03, 0.05]
episode_count = 100
strategy_list = ["shortest", "safe", "predictive"]
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def load_map(file_name):
    map_rows = []
    with open(file_name, "r", encoding="utf-8") as f:
        for text_line in f:
            text_line = text_line.strip()
            if not text_line:
                continue
            cells = text_line.split() if " " in text_line or "\t" in text_line else list(text_line)
            map_rows.append([int(cell) for cell in cells])

    field = np.array(map_rows, dtype=int)

    if field.ndim != 2 or field.shape[0] < 50 or field.shape[1] < 50:
        raise ValueError("размер карты должен быть не меньше 50x50")

    if not set(np.unique(field)).issubset({0, 1, 2, 3, 4}):
        raise ValueError("в карте допустимы только значения 0, 1, 2, 3, 4")

    start_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 2)]
    spawn_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 3)]
    goal_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 4)]

    if len(start_cells) != 3:
        raise ValueError("на карте должно быть ровно три стартовые клетки 2")
    if not spawn_cells:
        raise ValueError("на карте должна быть хотя бы одна клетка 3")
    if not goal_cells:
        raise ValueError("на карте должна быть хотя бы одна клетка 4")

    return field, start_cells, spawn_cells, goal_cells

# проверка доступности клетки
def passable(field, place):
    row, col = place
    return 0 <= row < field.shape[0] and 0 <= col < field.shape[1] and field[row, col] != 1

# доступные соседние клетки
def neighbors(field, place):
    row, col = place
    next_cells = []
    for row_step, col_step in moves:
        next_place = (row + row_step, col + col_step)
        if passable(field, next_place):
            next_cells.append(next_place)
    return next_cells

# подсчет расстояния от клеток до цели
def distance_map(field, goal_cells):
    big_num = 10 ** 9
    distance = np.full(field.shape, big_num, dtype=int)
    queue = deque()

    for target in goal_cells:
        distance[target] = 0
        queue.append(target)

    while queue:
        current = queue.popleft()
        for next_place in neighbors(field, current):
            if distance[next_place] == big_num:
                distance[next_place] = distance[current] + 1
                queue.append(next_place)

    return distance

def random_choice(variants):
    return variants[int(random.random() * len(variants))]

# ход машины
def vehicle_next(field, car):
    place = car["pos"]
    possible = neighbors(field, place)

    if not possible:
        return place

    # единственный выход для тупиков
    if len(possible) == 1:
        return possible[0]

    last_dir = car["dir"]

    if last_dir is None:
        return random_choice(possible)

    if len(possible) >= 3:
        return random_choice(possible)

    back_place = (place[0] - last_dir[0], place[1] - last_dir[1])
    forward_place = (place[0] + last_dir[0], place[1] + last_dir[1])

    without_back = [p for p in possible if p != back_place]

    if forward_place in without_back and random.random() < 0.7:
        return forward_place

    if without_back:
        return random_choice(without_back)

    return place

# спавн машин
def spawn_vehicles(spawn_cells, cars, player_place, spawn_chance):
    busy_cells = {v["pos"] for v in cars}

    # проверка каждой точки
    for place in spawn_cells:
        if random.random() < spawn_chance and place not in busy_cells:
            if place == player_place:
                return True

            # жизнь машины
            life = 15 + int(random.random() * 136)
            cars.append({"pos": place, "dir": None, "ttl": life})
            busy_cells.add(place)

    return False

# двигаем машины на один шаг
def move_vehicles(field, cars, player_place):
    hit = False

    for car in cars:
        old_place = car["pos"]
        new_place = vehicle_next(field, car)

        # обнова направления движения
        if new_place != old_place:
            car["dir"] = (new_place[0] - old_place[0], new_place[1] - old_place[1])

        car["pos"] = new_place
        car["ttl"] -= 1

        if new_place == player_place:
            hit = True

    # убираем мертвые
    cars[:] = [v for v in cars if v["ttl"] > 0]
    return hit

# выбор действия агента
def choose_agent_move(field, distance, player_place, cars, method):
    busy_cells = {v["pos"] for v in cars}
    possible = neighbors(field, player_place)

    # стоп машины
    if random.random() < 0.05:
        return player_place

    # кратчайший
    if method == "shortest":
        return min(possible, key=lambda p: distance[p])

    # оставляем свободные клетки для движения
    free_cells = [p for p in possible if p not in busy_cells]

    # безопасная стратегия
    if method == "safe":
        return min(free_cells, key=lambda p: distance[p]) if free_cells else player_place

    # опасные клетки
    danger_cells = set(busy_cells)
    for car in cars:
        danger_cells.update(neighbors(field, car["pos"]))

    # безопасные клетки
    safe_cells = [p for p in free_cells if p not in danger_cells]
    if safe_cells:
        return min(safe_cells, key=lambda p: distance[p])
    if free_cells:
        return min(free_cells, key=lambda p: distance[p])
    return player_place

# один эпизод моделирования
def run_episode(field, spawn_cells, goal_cells, distance, start, method, spawn_chance, step_limit, save_frames=False):
    player = start
    cars = []
    frames = [(player, [])] if save_frames else None
    shortest_path = int(distance[start])

    for step_num in range(1, step_limit + 1):
        if spawn_vehicles(spawn_cells, cars, player, spawn_chance):
            return False, step_num, 0.0, frames

        new_player = choose_agent_move(field, distance, player, cars, method)

        if new_player in {v["pos"] for v in cars}:
            return False, step_num, 0.0, frames

        player = new_player

        if player in goal_cells:
            if save_frames:
                frames.append((player, [v["pos"] for v in cars]))
            return True, step_num, shortest_path / step_num, frames

        if move_vehicles(field, cars, player):
            if save_frames:
                frames.append((player, [v["pos"] for v in cars]))
            return False, step_num, 0.0, frames

        if save_frames:
            frames.append((player, [v["pos"] for v in cars]))

    return False, step_limit, 0.0, frames

# серия экспериментов
def run_experiments(field, start_cells, spawn_cells, goal_cells, distance):
    map_rows = []

    for seed in seed_list:
        for spawn_chance in spawn_chances:
            for start_id, start in enumerate(start_cells, 1):
                shortest_path = int(distance[start])
                if shortest_path >= 10 ** 9:
                    raise ValueError(f"из старта {start_id} нет пути до цели")

                # макс длина эпизода
                step_limit = max(300, shortest_path * 8)

                for method in strategy_list:

                    random.seed(seed)

                    for episode in range(episode_count):
                        is_success, steps, delivery_speed, _ = run_episode(
                            field, spawn_cells, goal_cells, distance, start,
                            method, spawn_chance, step_limit
                        )

                        map_rows.append({
                            "seed": seed,
                            "spawn_prob": spawn_chance,
                            "start": start_id,
                            "strategy": method,
                            "episode": episode + 1,
                            "success": int(is_success),
                            "steps": steps,
                            "speed": delivery_speed,
                        })

    return pd.DataFrame(map_rows)

def make_summary(results_df):
    work_df = results_df.copy()
    work_df["success_steps"] = np.where(work_df["success"] == 1, work_df["steps"], np.nan)
    work_df["success_speed"] = np.where(work_df["success"] == 1, work_df["speed"], np.nan)

    return (
        work_df.groupby(["seed", "spawn_prob", "start", "strategy"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_steps=("success_steps", "mean"),
            mean_speed=("success_speed", "mean"),
        )
    )

def save_map(field):
    cmap = ListedColormap(["white", "black", "deepskyblue", "orange", "green"])
    plt.figure(figsize=(7, 7))
    plt.imshow(field, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    plt.title("карта среды")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "citymap.png"), dpi=160)
    plt.close()

def save_mean_steps(summary_df):
    figure, plots = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for start, plot in enumerate(plots, 1):
        start_data = summary_df[summary_df["start"] == start]

        for method in strategy_list:
            plot_data = start_data[start_data["strategy"] == method]
            avg_values = plot_data.groupby("spawn_prob")["mean_steps"].mean()
            plot.plot(avg_values.index, avg_values.values, marker="o", label=method)

        plot.set_title(f"старт {start}")
        plot.set_xlabel("вероятность генерации")
        plot.grid(alpha=0.3)

    plots[0].set_ylabel("среднее число шагов")
    plots[-1].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(result_dir, "mean_steps.png"), dpi=160)
    plt.close(figure)

def save_boxplot(summary_df, value_column, axis_label, file_name):
    figure, plots = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    for start, plot in enumerate(plots, 1):
        start_data = summary_df[summary_df["start"] == start]
        plot_data = []

        for method in strategy_list:
            values = start_data[
                start_data["strategy"] == method
                ][value_column].dropna().values
            plot_data.append(values)

        plot.boxplot(plot_data)
        plot.set_xticks(range(1, len(strategy_list) + 1))
        plot.set_xticklabels(strategy_list)
        plot.set_title(f"старт {start}")
        plot.set_xlabel("стратегия")
        plot.grid(axis="y", alpha=0.3)

    plots[0].set_ylabel(axis_label)
    figure.tight_layout()
    figure.savefig(os.path.join(result_dir, file_name), dpi=160)
    plt.close(figure)

def save_animation(field, start_cells, spawn_cells, goal_cells, distance):
    random.seed(seed_list[0])
    start = start_cells[0]
    step_limit = max(300, int(distance[start]) * 8)

    _, _, _, frames = run_episode(
        field, spawn_cells, goal_cells, distance, start,
        "predictive", 0.02, step_limit, save_frames=True
    )

    cmap = ListedColormap([
        "white", "black", "deepskyblue", "orange",
        "green", "red", "blue"
    ])

    def frame_array(player, cars):
        frame = field.copy()
        for place in cars:
            frame[place] = 5
        frame[player] = 6
        return frame

    figure, plot = plt.subplots(figsize=(7, 7))
    map_image = plot.imshow(frame_array(*frames[0]), cmap=cmap, vmin=0, vmax=6)
    plot_title = plot.set_title("шаг 0")

    def update(frame_num):
        map_image.set_data(frame_array(*frames[frame_num]))
        plot_title.set_text(f"шаг {frame_num}")
        return map_image, plot_title

    animation = FuncAnimation(figure, update, frames=len(frames), interval=120, blit=False)
    animation.save(os.path.join(result_dir, "agent.gif"), writer=PillowWriter(fps=8))
    plt.close(figure)

def main():
    os.makedirs(result_dir, exist_ok=True)

    field, start_cells, spawn_cells, goal_cells = load_map(map_name)
    distance = distance_map(field, goal_cells)

    results_df = run_experiments(field, start_cells, spawn_cells, goal_cells, distance)
    summary_df = make_summary(results_df)

    results_df.to_csv(
        os.path.join(result_dir, "results.csv"),
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        os.path.join(result_dir, "summary.csv"),
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    )

    save_map(field)
    save_mean_steps(summary_df)
    save_boxplot(summary_df, "mean_speed", "скорость доставки", "boxplot_speed.png")
    save_boxplot(summary_df, "success_rate", "вероятность успеха", "boxplot_success.png")
    save_animation(field, start_cells, spawn_cells, goal_cells, distance)

    print("готово")
    print("результаты сохранены в папку results")

if __name__ == "__main__":
    main()