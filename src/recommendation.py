import numpy as np
from typing import List, Tuple, Dict
from numpy.typing import NDArray


SongScore = Tuple[str, float]

def get_similar_songs(query_vector: NDArray, db: Dict[str, NDArray]) -> List[SongScore]: # type: ignore
    results = []

    for song, vector in db.items():
        score = np.linalg.norm(query_vector - vector)

        results.append((song, score))

        results.sort(key=lambda x: x[1])


        if len(results) <= 1:
            return results
        filtered = results[1:]

        if not filtered:
            return results
        return filtered[:3]