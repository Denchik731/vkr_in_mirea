# e.py
# ---------------------------
# Интерактивная визуализация + анализ значимости скрытых сообществ
# Работает с edge-list форматом:
#   edges_df: user_id ; community_id
#   topics_csv: community_id ; topic ; name
# ---------------------------


"""
                                          Пояснялка для памяти,что я тут писал:


Общая идея этого файла e.py

Gecnm у нас есть граф с несколькими сообществами. Мы хотим найти наиболее значимые из них. Вот как это работает:

1.Поиск сообществ:
    Используем алгоритм Louvain для поиска сообществ в графе.
2.Анализ сообществ:
    Для каждого сообщества вызываем функцию _community_subgraph_metrics, чтобы вычислить его метрики.
3.Сортировка:
    Сортируем сообщества по значимости (significance_score).
4.Визуализация:
    Отображаем граф с выделенными сообществами и их метриками.



Методы библиотеки networks:
    +degree -  принимает вершину и возрращает кол-во ее ребер( степень вершины)
    +subgraph -  принимает список узлов и создает граф( принимает обязательно итерируемы объект!!!)

    score = density * avg_wdeg * math.log(n + 1) # эта строка вычисляет значимость подграфа (score), которая используется
                                                                          для оценки важности скрытого сообщества в графе
    * density — плотность подграфа это отношение количества рёбер в подграфе к максимальному возможному количеству рёбер
    * avg_wdeg — средняя взвешенная степень (Средняя взвешенная степень — это сумма весов всех рёбер, соединяющих узел с
                                              другими узлами, деленная на количество узлов.)
    * math.log(n + 1) — логарифм размера подграфа -
                        Логарифм размера подграфа используется для учета размера сообщества. Чем больше размер, тем больше вклад в значимость.
                        Логарифм используется для нормализации размера, чтобы большие сообщества не доминировали над маленькими.
   параметр weight - вес ребра, этот вес определят как сильно связаны два наших аккаунта вк

   *avg_internal_weighted_degree — средняя взвешенная степень:
                                    Средняя взвешенная степень — это сумма весов всех рёбер,
                                    соединяющих узел с другими узлами, деленная на количество узлов.

"""


from __future__ import annotations # это просто для анотаций нафиг не надо было мне это ))

import math
from collections import Counter, defaultdict # для работы с коллекцияим
from typing import Dict, List, Tuple # для анотации типов(удобно)

import networkx as nx #
import plotly.graph_objects as go # для создания графика
import pandas as pd
from community import community_louvain  # лувенкий метод


# ---------------------------
# Загрузка тематик сообществ
# ---------------------------

def load_topics_maps(topics_csv_path: str) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Загружаем community_topics.csv (community_id;topic;name)
    Возвращаем:
      - topic_map: community_id -> topic
      - name_map : community_id -> name (для красоты в hover/панели)
    """
    df = pd.read_csv(topics_csv_path, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required = {"community_id", "topic"}
    if not required.issubset(df.columns):
        raise ValueError(f"topics CSV должен иметь колонки {required}. Есть: {list(df.columns)}")

    topic_map = dict(zip(df["community_id"].astype(str), df["topic"].astype(str)))

    if "name" in df.columns:
        name_map = dict(zip(df["community_id"].astype(str), df["name"].astype(str)))
    else:
        name_map = {}

    return topic_map, name_map


# ---------------------------
# Метрики подграфа (значимость скрытого сообщества)
# ---------------------------

def _community_subgraph_metrics(G: nx.Graph, nodes: List[str], weight: str = "weight") -> dict:
    """
   Принимает параметры: G - объект класса граф, nodes - список узлов напимер ['A','B','C'], weight- вес ребра
   Возращает: словарь с метриками подграфа

    """
    sub = G.subgraph(nodes) #  создаем подграф графа G, переданного в функцию --> просто метод из библы граф
    n = sub.number_of_nodes() # возращаем кол-во узлов в графе
    m = sub.number_of_edges() # возращаем кол-во ребер в графе

    max_edges = n * (n - 1) / 2 if n > 1 else 1 # вычисляется максимальное возможное количество рёбер в подграфе
    density = (m / max_edges) if max_edges > 0 else 0.0 # вычисляется плотность подграфа (density), которая равна отношению количества рёбер к максимальному возможному количеству рёбер.

    wdeg = dict(sub.degree(weight=weight)) # вычисляется взвешенная степень каждого узла
    avg_wdeg = (sum(wdeg.values()) / n) if n > 0 else 0.0 # вычисляется средняя взвешенная степень (avg_wdeg), которая равна сумме всех взвешенных степеней, деленной на количество узлов.

    score = density * avg_wdeg * math.log(n + 1) # значимость подграфа( у нас это значимость скрытого со-ва вк)

    # метрики скрытого сооб-ва вк
    return {
        "size": n, # количество узлов в подграфе( акаунты)
        "edges": m,# кол-во ребер
        "density": density,# плотность
        "avg_internal_weighted_degree": avg_wdeg,# средняя взвешенная степень подграфа
        "significance_score": score,# значимость этого подграфа( соо-ва людей)
    }


# ---------------------------
# Вспомогательные функции "что объединяет"
# ---------------------------

def build_user_to_groups_from_edges(edges_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Функция build_user_to_groups_from_edges создает словарь, где ключами являются user_id,
     а значениями — списки community_id, к которым принадлежит пользователь.
    """
    user_to_groups: Dict[str, List[str]] = defaultdict(list)
    for _, r in edges_df.iterrows():
        user_to_groups[str(r["user_id"])].append(str(r["community_id"]))
    return user_to_groups

#  находит топ-n сообществ, к которым принадлежат пользователи в кластере
def _top_groups_inside_cluster(user_to_groups: Dict[str, List[str]],cluster_users: List[str],top_n: int = 5) -> List[Tuple[str, int]]:
    all_groups = [] # все сообщества, к которым принадлежат поль-ли в данном кластере(подграфе)
    for uid in cluster_users: # идем по списку поль-лей кластера
        all_groups.extend(user_to_groups.get(uid, []))# список сообществ для пользователя uid
    return Counter(all_groups).most_common(top_n)# ко-во вхождений сообществ
    #возвращаем список кортежей, где каждый кортеж содержит community_id и количество его вхождений.



# находит топ-n тематик, которые встречаются в кластере.
def _top_topics_inside_cluster(user_to_groups: Dict[str, List[str]],cluster_users: List[str],topic_map: Dict[str, str],top_n: int = 5) -> List[Tuple[str, int]]:
    topics = []
    for uid in cluster_users:
        for gid in user_to_groups.get(uid, []):
            t = topic_map.get(str(gid))
            if t:
                topics.append(t)
    return Counter(topics).most_common(top_n)


# ---------------------------
# Анализ скрытых сообществ (таблица + информация для hover)
# ---------------------------
#
def analyze_hidden_communities(G: nx.Graph, partition: Dict[str, int],user_to_groups: Dict[str, List[str]],topic_map: Dict[str, str],name_map: Dict[str, str],top_n_groups: int = 5):
    """
    Принимает: G: граф,
                partition: словарь, где ключами являются id пользователей, а значениями — id сообществ, к которым они принадлежат.
                user_to_groups: словарь, где ключами являются id пользователей, а значениями — списки id сообществ, к которым они принадлежат.
                topic_map: словарь, где ключами являются id сообществ, а значениями — их тематики.
                name_map: словарь, где ключами являются id сообществ, а значениями — их имена.
                top_n_groups: количество топ-сообществ, которые нужно найти (по умолчанию 5).
    Возвращает:
      - summary_rows: список строк (для панели и вывода)
      - cluster_info: cluster_id -> метрики и строки
    """

    clusters = defaultdict(list) # пустой пока
    # cоздается словарь clusters, где ключами являются id сообществ, а значениями — списки id пользователей, принадлежащих к этим сообществам.
    for uid, cid in partition.items():
        clusters[cid].append(uid)

    cluster_info = {}
    summary_rows = []

    # Для каждого сообщества (cid) вычисляются метрики подграфа с помощью функции _community_subgraph_metrics.
    # met — это словарь с метриками, такими как размер, плотность, средняя взвешенная степень и значимость.
    for cid, users in clusters.items():
        met = _community_subgraph_metrics(G, users, weight="weight")



        # Вычисляются топ-n сообществ внутри кластера с помощью функции _top_groups_inside_cluster.
        # Формируется строка top_groups_str, которая содержит информацию о топ-сообществах, включая их идентификаторы, количество вхождений и имена (если они есть)
        top_groups = _top_groups_inside_cluster(user_to_groups, users, top_n=top_n_groups)
        # красиво: community_id (count) + (name) если есть


        top_groups_str_parts = []    # тут будет списко строк- инфы о топ 5 сообществах


        for g, c in top_groups:   # идем по списку сообществ, g — id сообщества (community_id),c — количество вхождений сообщества.
            nm = name_map.get(str(g)) #  получаем имя сообщества из словаря name_map. Если имя не найдено, возвращается None.
            if nm:
                top_groups_str_parts.append(f"{g} ({c}) — {nm}")
            else:
                top_groups_str_parts.append(f"{g} ({c})")
        top_groups_str = "<br>".join(top_groups_str_parts) if top_groups_str_parts else "нет данных"

        top_topics = _top_topics_inside_cluster(user_to_groups, users, topic_map, top_n=5)
        top_topics_str = ", ".join([f"{t} ({c})" for t, c in top_topics]) if top_topics else "нет данных"

        cluster_info[cid] = { # # Создаем словарь для текущего кластера (hidden community)
            **met, # Распаковываем метрики подграфа (size, density, score
            "top_groups": top_groups, #Сохраняем сырые данные ТОП-5 групп
            "top_groups_str": top_groups_str,#
            "top_topics": top_topics,#
            "top_topics_str": top_topics_str,#
        }

        summary_rows.append({     # Добавляем строку в таблицу ТОП-сообществ (список словарей)
            "hidden_comm_id": cid,
            "size_users": met["size"], # Количество пользователей в сообществе
            "density": round(met["density"], 4), #Плотность связей (0.0-1.0) округляю до 4 знаков полсе запятой
            "avg_wdeg": round(met["avg_internal_weighted_degree"], 4), # Средняя.взв.степень внутри кластера
            "score": round(met["significance_score"], 6), # ГЛАВНАЯ метрика значимости density × avg_wdeg × log(размер+1)
            "top_topics": top_topics_str, # ТОП-5 тематик
            "обобщающий_признак": top_groups_str.replace("<br>", "; "), # Преобразует HTML
        })


    # сортировка списка словарей по значимости
    summary_rows.sort(key=lambda x: x["score"], reverse=True)



    return summary_rows, cluster_info


# ---------------------------
# Основная визуализация
# ---------------------------

def visualize_network_advanced(
    G: nx.Graph,
    edges_df: pd.DataFrame,
    topics_csv_path: str,
    title: str = "Анализ скрытых сообществ ВКонтакте",
    show: bool = True,
    max_nodes_plot: int = 2500
):
    """
    Интерактивная визуализация:

    """

    if G.number_of_edges() == 0: # если ребер нет - граф пустой выкину исключение
        print("ERRRR")
        raise ValueError(
            "Граф получился без рёбер.\n"
            "Проверьте параметры build_similarity_graph:\n"
            "- уменьшить threshold (например 0.15)\n"
            "- увеличиить k_neighbors (например 50)\n"
            "Также проверьте, что communities не пустые."
        )

    # Тематики и имена сообществ
    topic_map, name_map = load_topics_maps(topics_csv_path)

    # Лувенкский алгоритм
    partition = community_louvain.best_partition(G, weight="weight")

    # Модулярность - функционал качетсва разбиения графа на сообщества



    modularity = community_louvain.modularity(partition, G, weight="weight")

    # user_id -> list[community_id]
    user_to_groups = build_user_to_groups_from_edges(edges_df)

    # Анализ значимости
    summary_rows, cluster_info = analyze_hidden_communities(
        G, partition, user_to_groups, topic_map, name_map, top_n_groups=5
    )

    # Ограничим количество узлов для Plotly (иначе тяжело)
    nodes_all = list(G.nodes())
    if len(nodes_all) > max_nodes_plot:
        nodes_plot = nodes_all[:max_nodes_plot]
        H = G.subgraph(nodes_plot).copy()
    else:
        H = G

    # Layout
    pos = nx.spring_layout(H, k=0.7, iterations=30, weight="weight", seed=42)

    # Узлы
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    degrees = dict(H.degree())
    max_deg = max(degrees.values()) if degrees else 1

    for uid in H.nodes():
        x, y = pos[uid]
        node_x.append(x)
        node_y.append(y)

        cid = partition.get(uid, -1)
        info = cluster_info.get(cid, {})

        text = (
            f"<b>user_id:</b> {uid}<br>"
            f"<b>hidden_comm_id:</b> {cid}<br>"
            f"<b>размер скрытого сообщества:</b> {info.get('size', 0)}<br>"
            f"<b>score:</b> {info.get('significance_score', 0):.4f}<br>"
            f"<b>тематики (ТОП):</b> {info.get('top_topics_str', 'нет данных')}<br><br>"
            f"<b>сообщества (ТОП):</b><br>{info.get('top_groups_str', 'нет данных')}"
        )
        node_text.append(text)

        node_color.append(cid)
        node_size.append(6 + 18 * (degrees.get(uid, 0) / max_deg))

    # Рёбра
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",  #ребра
        line=dict(width=1, color="rgba(0,255,130,0.18)"),
        hoverinfo="none",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            line=dict(width=1, color="rgba(255,255,255,0.70)"),
            opacity=0.95
        ),
        hovertemplate="%{text}<extra></extra>",
        text=node_text,
        showlegend=False
    ))

    # Панель справа: ТОП скрытых сообществ
    top_lines = []
    for r in summary_rows[:5]:
        top_lines.append(
            f"• <b>ID {r['hidden_comm_id']}</b>: score={r['score']} | size={r['size_users']}<br>"
            f"&nbsp;&nbsp;<b>тематики:</b> {r['top_topics']}<br>"
            f"&nbsp;&nbsp;<b>признак:</b> {r['обобщающий_признак']}<br><br>"
        )

    panel_text = "<b>ТОП скрытых сообществ (по значимости)</b><br><br>" + "".join(top_lines)

    fig.add_annotation( # леганда панель
        xref="paper", yref="paper",
        #x=1.08, y=0.98,
        x=1.01, y=0.98,
        xanchor="left", yanchor="top",
        text=panel_text,
        showarrow=False,
        align="left",
        font=dict(color="white", size=12),
        bgcolor="rgba(20,26,34,0.92)",
        bordercolor="rgba(168,85,247,0.7)",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><span style='font-size:14px;'>Модулярность: {modularity:.4f}</span>",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            font=dict(color="white", size=22)
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="white"),
        margin=dict(l=40, r=420, t=80, b=40),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        width=1600,
        height=820,
    )

    if show:
        fig.show()

    return partition, summary_rows, cluster_info, fig
