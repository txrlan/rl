import os
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

PROB_STRAIGHT = 0.7
MIN_CAR_TTL = 15
MAX_CAR_TTL = 150
MIN_EPISODE_STEPS = 300
LIMIT_MULTIPLIER = 8

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

            if " " in text_line or "\t" in text_line:
                cells = text_line.split()
            else:
                cells = list(text_line)

            map_rows.append([int(cell) for cell in cells])

    field = np.array(map_rows, dtype=int)
    if field.ndim != 2 or field.shape[0] < 50 or field.shape[1] < 50:
        raise ValueError("размер карты должен быть не меньше 50x50")

    start_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 2)]
    spawn_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 3)]
    goal_cells = [tuple(map(int, cell)) for cell in np.argwhere(field == 4)]

    return field, start_cells, spawn_cells, goal_cells

def passable(field, place):
    row, col = place
    return 0 <= row < field.shape[0] and 0 <= col < field.shape[1] and field[row, col] != 1

def neighbors(field, place):
    row, col = place
    next_cells = []
    for r, c in moves:
        next_place = (row + r, col + c)
        if passable(field, next_place):
            next_cells.append(next_place)
    return next_cells

def distance_map(field, goal_cells):
    big_num = 10 ** 9
    distance = np.full(field.shape, big_num, dtype=int)
    queue = deque(goal_cells)

    for target in goal_cells:
        distance[target] = 0

    while queue:
        current = queue.popleft()
        for next_place in neighbors(field, current):
            if distance[next_place] == big_num:
                distance[next_place] = distance[current] + 1
                queue.append(next_place)

    return distance

def vehicle_next(field, car):
    place = car["pos"]
    possible = neighbors(field, place)

    if not possible:
        return place
    if len(possible) == 1:
        return possible[0]

    last_dir = car["dir"]
    if last_dir is None:
        return random.choice(possible)

    forward_place = (place[0] + last_dir[0], place[1] + last_dir[1])

    if forward_place in possible and random.random() < PROB_STRAIGHT:
        return forward_place

    others = [p for p in possible if p != forward_place]
    if others:
        return random.choice(others)
    else:
        return place

def spawn_vehicles(spawn_cells, cars, player_place, spawn_chance):
    busy_cells = {v["pos"] for v in cars}
    hit = False

    for place in spawn_cells:
        if random.random() < spawn_chance and place not in busy_cells:
            if place == player_place:
                hit = True

            cars.append({
                "pos": place,
                "dir": None,
                "ttl": random.randint(MIN_CAR_TTL, MAX_CAR_TTL)
            })
            busy_cells.add(place)

    return hit

def move_vehicles(field, cars, player_place):
    hit = False
    for car in cars:
        new_place = vehicle_next(field, car)
        if new_place != car["pos"]:
            car["dir"] = (new_place[0] - car["pos"][0], new_place[1] - car["pos"][1])

        car["pos"] = new_place
        car["ttl"] -= 1

        if new_place == player_place:
            hit = True

    cars[:] = [v for v in cars if v["ttl"] > 0]
    return hit

def choose_agent_move(field, distance, player_place, cars, method):
    possible = neighbors(field, player_place) + [player_place]
    if method == "shortest":
        return min(possible, key=lambda p: distance[p])

    busy_cells = {v["pos"] for v in cars}
    free_cells = [p for p in possible if p not in busy_cells]

    if method == "safe":
        if free_cells:
            return min(free_cells, key=lambda p: distance[p])
        else:
            return player_place

    danger_cells = set(busy_cells)
    for car in cars:
        danger_cells.update(neighbors(field, car["pos"]))

    safe_cells = [p for p in free_cells if p not in danger_cells]

    if safe_cells:
        return min(safe_cells, key=lambda p: distance[p])

    if free_cells:
        return min(free_cells, key=lambda p: distance[p])

    return player_place

def run_episode(field, spawn_cells, goal_cells, distance, start, method, spawn_chance, step_limit, save_frames=False):
    player = start
    cars = []

    if save_frames:
        frames = [(player, [])]
    else:
        frames = None

    shortest_path = int(distance[start])

    for step_num in range(1, step_limit + 1):
        if spawn_vehicles(spawn_cells, cars, player, spawn_chance):
            return False, step_num, np.nan, frames

        player = choose_agent_move(field, distance, player, cars, method)

        if player in {v["pos"] for v in cars}:
            return False, step_num, np.nan, frames

        if move_vehicles(field, cars, player):
            if save_frames:
                frames.append((player, [v["pos"] for v in cars]))
            return False, step_num, np.nan, frames

        if save_frames:
            frames.append((player, [v["pos"] for v in cars]))

        if player in goal_cells:
            return True, step_num, shortest_path / step_num, frames

    return False, step_limit, np.nan, frames

def run_experiments(field, start_cells, spawn_cells, goal_cells, distance):
    map_rows = []
    for seed in seed_list:
        for spawn_chance in spawn_chances:
            for start_id, start in enumerate(start_cells, 1):
                shortest_path = int(distance[start])
                if shortest_path >= 10 ** 9:
                    continue

                step_limit = max(MIN_EPISODE_STEPS, shortest_path * LIMIT_MULTIPLIER)

                for method in strategy_list:
                    random.seed(seed)
                    for episode in range(episode_count):
                        is_success, steps, speed, _ = run_episode(
                            field, spawn_cells, goal_cells, distance, start, method, spawn_chance, step_limit
                        )

                        map_rows.append({
                            "seed": seed,
                            "spawn_prob": spawn_chance,
                            "start": start_id,
                            "strategy": method,
                            "episode": episode + 1,
                            "success": int(is_success),
                            "steps": steps,
                            "speed": speed
                        })

    return pd.DataFrame(map_rows)

def make_summary(results_df):
    return results_df.groupby(
        ["seed", "spawn_prob", "start", "strategy"],
        as_index=False
    ).agg(
        success_rate=("success", "mean"),
        mean_steps=("steps", "mean"),
        mean_speed=("speed", "mean")
    )

def save_map(field):
    plt.figure(figsize=(7, 7))
    plt.imshow(field, cmap=ListedColormap(["white", "black", "deepskyblue", "orange", "green"]), vmin=0, vmax=4)
    plt.title("карта среды");
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "citymap.png"), dpi=160);
    plt.close()

def save_mean_steps(summary_df):
    figure, plots = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for start, plot in enumerate(plots, 1):
        start_data = summary_df[summary_df["start"] == start]
        for method in strategy_list:
            plot_data = start_data[start_data["strategy"] == method]
            avg_values = plot_data.groupby("spawn_prob")["mean_steps"].mean()
            plot.plot(avg_values.index, avg_values.values, marker="o", label=method)
        plot.set_title(f"старт {start}");
        plot.set_xlabel("вероятность генерации");
        plot.grid(alpha=0.3)
    plots[0].set_ylabel("среднее число шагов");
    plots[-1].legend()
    figure.tight_layout();
    figure.savefig(os.path.join(result_dir, "mean_steps.png"), dpi=160);
    plt.close(figure)

def save_boxplot(df, value_column, axis_label, file_name):
    figure, plots = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for start, plot in enumerate(plots, 1):
        plot_data = [df[(df["start"] == start) & (df["strategy"] == method)][value_column].dropna().values
                     for method in strategy_list]
        plot.boxplot(plot_data)
        plot.set_xticks(range(1, len(strategy_list) + 1));
        plot.set_xticklabels(strategy_list)
        plot.set_title(f"старт {start}");
        plot.set_xlabel("стратегия");
        plot.grid(axis="y", alpha=0.3)
    plots[0].set_ylabel(axis_label)
    figure.tight_layout();
    figure.savefig(os.path.join(result_dir, file_name), dpi=160);
    plt.close(figure)

def save_animation(field, start_cells, spawn_cells, goal_cells, distance):
    anim_seed = seed_list[0]
    anim_start = start_cells[0]
    anim_strategy = "predictive"
    anim_spawn_prob = 0.02

    random.seed(anim_seed)
    _, _, _, frames = run_episode(
        field, spawn_cells, goal_cells, distance, anim_start, anim_strategy, anim_spawn_prob,
        max(MIN_EPISODE_STEPS, int(distance[anim_start]) * LIMIT_MULTIPLIER), save_frames=True
    )

    cmap = ListedColormap(["white", "black", "deepskyblue", "orange", "green", "red", "blue"])

    def frame_array(player, cars):
        frame = field.copy()
        for place in cars: frame[place] = 5
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
    animation.save(os.path.join(result_dir, "agent.gif"), writer=PillowWriter(fps=8));
    plt.close(figure)

def main():
    os.makedirs(result_dir, exist_ok=True)
    field, start_cells, spawn_cells, goal_cells = load_map(map_name)
    distance = distance_map(field, goal_cells)

    results_df = run_experiments(field, start_cells, spawn_cells, goal_cells, distance)
    summary_df = make_summary(results_df)

    results_df.to_csv(os.path.join(result_dir, "results.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(result_dir, "summary.csv"), index=False, sep=";", decimal=",", encoding="utf-8-sig")

    save_map(field)
    save_mean_steps(summary_df)
    save_boxplot(results_df, "speed", "скорость доставки", "boxplot_speed.png")
    save_boxplot(summary_df, "success_rate", "вероятность успеха", "boxplot_success.png")
    save_animation(field, start_cells, spawn_cells, goal_cells, distance)

    print("готово\nрезультаты сохранены в папку results")

if __name__ == "__main__":
    main()