# create_ug_matrix.py
# -------------------------------------------------
# Создание sparse-матрицы user × community из edge-list CSV:
#   user_id ; community_id
# -------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from scipy.sparse import csr_matrix   # разреженная матрица


@dataclass(frozen=True) # гарантирует что матрица не поменяетс, кешируемость
class UserCommunityData:

    csr: csr_matrix # матрица (n_users × n_communities), значения 0/1
    user_ids: List[str]# порядок строк
    community_ids: List[str]# порядок столбцов
    user_index: Dict[str, int]#
    comm_index: Dict[str, int]#
    edges_df: pd.DataFrame#

    @classmethod
    def from_edges_df(cls, edges_df: pd.DataFrame) -> UserCommunityData:
        """
        Строит разреженную матрицу user × community из edge-list таблицы.


        """

        if "user_id" not in edges_df.columns or "community_id" not in edges_df.columns:
            raise ValueError(
                f"Нужны колонки user_id и community_id. Сейчас: {list(edges_df.columns)}"
            )

        # Нормализация типов и пробелов
        df = edges_df[["user_id", "community_id"]].copy()
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["community_id"] = df["community_id"].astype(str).str.strip()

        # Уникальные пользователи и сообщества (порядок фиксируем через drop_duplicates)
        user_ids = df["user_id"].drop_duplicates().tolist()
        community_ids = df["community_id"].drop_duplicates().tolist()

        # Индексы
        user_index = {u: i for i, u in enumerate(user_ids)}
        comm_index = {c: j for j, c in enumerate(community_ids)}

        # Координаты единиц
        row_idx = df["user_id"].map(user_index).to_numpy()
        col_idx = df["community_id"].map(comm_index).to_numpy()

        data = [1] * len(df)

        # Sparse CSR матрица (очень быстрая и экономная)
        csr = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(user_ids), len(community_ids)),
            dtype=int
        )

        return cls(
            csr=csr,
            user_ids=user_ids,
            community_ids=community_ids,
            user_index=user_index,
            comm_index=comm_index,
            edges_df=df
        )