from dataclasses import dataclass, field
from .graph_generator import Generation
from urllib.request import urlretrieve
from collections import defaultdict
from ..logs.logs import Logger
from zipfile import ZipFile
from functools import wraps
from tqdm import tqdm
import pandas as pd
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Utility(Generation):
    movies: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False, compare=False)
    full_data_x: np.ndarray = field(init=False, default_factory=lambda: np.ndarray, repr=False, compare=False)
    full_data_y: np.ndarray = field(init=False, default_factory=lambda: np.ndarray, repr=False, compare=False)
    voc_dict: dict = field(init=False, default_factory=dict, repr=False, compare=False)
    voc_dict_reverse: dict = field(init=False, default_factory=dict, repr=False, compare=False)
    batch_size: int = field(init=True, default=int, repr=False, compare=False)
    total_batches: int = field(init=False, default=int, repr=False, compare=False)
    logger: object = field(init=False, default_factory=object, repr=False, compare=False)

    def getLogger(self) -> object:
        return self.logger

    def __post_init__(self):
        logger = Logger.loggerSetup()
        object.__setattr__(self, 'logger', logger)

        full_data_x, full_data_y, voc_dict, voc_dict_reverse, movies = self.getGraphData()
        num_data = len(full_data_x)
        total_batches = num_data // self.batch_size + 1 if num_data % self.batch_size > 0 else num_data // self.batch_size

        object.__setattr__(self, 'full_data_x', np.array(full_data_x))
        object.__setattr__(self, 'full_data_y', np.array(full_data_y))
        object.__setattr__(self, 'voc_dict_reverse', voc_dict_reverse)
        object.__setattr__(self, 'total_batches', total_batches)
        object.__setattr__(self, 'voc_dict', voc_dict)
        object.__setattr__(self, 'movies', movies)

    def get_movie_id_by_title(self, title: str) -> list:
        return list(self.movies[self.movies.title == title].movieId)[0]

    def getValidExamples(self, valid_word: list) -> list:
        valid_examples = []

        for movie_title in valid_word:
            movieId = self.get_movie_id_by_title(movie_title)
            token_id = self.voc_dict[movieId]
            valid_examples.append(token_id)

        return valid_examples

    def _createTrainingData(func: object) -> object:
        """
        Wrapper in order to transform movie data into training
        node based data.
        :return: object wrapper
        """

        @wraps(func)
        def createTrainingData(self, *args, **kwargs) -> tuple:
            """
            Get data vector mapping
            :return: data dictionary for vector mapping
            """

            # Set logger.
            logger = self.getLogger()

            # Get movies and rating.
            rated_movies, movies = func(self, *args, **kwargs)

            # Mapping dictionary of movies as nodes.
            data_dict_vec = defaultdict(set)

            # Group instances by user.
            movies_grouped_by_users = list(rated_movies.groupby("userId"))

            for group in tqdm(
                    movies_grouped_by_users,
                    position=0,
                    leave=True,
                    desc="Compute movie rating frequencies",
            ):
                # Get a list of movies rated by the user.
                current_movies = list(group[1]["movieId"])

                for i in range(len(current_movies)):
                    for j in range(i + 1, len(current_movies)):
                        data_dict_vec[current_movies[i]].add(current_movies[j])
                        data_dict_vec[current_movies[j]].add(current_movies[i])

            Logger.info(Logger().getLogDictInfo('createTrainingData', __name__, 'createTrainingData'),
                        'Populated data_dict_vec dictionary.', logger)

            # Save Vocab with retreivable indices for later.
            movie_dict = dict(zip(movies['movieId'], movies['title']))
            voc_dict = {x: i for i, x in enumerate(list(data_dict_vec.keys()))}
            voc_dict_reverse = {i: movie_dict[x] for i, x in enumerate(list(data_dict_vec.keys()))}

            # Convert data dic to dic of numeric.
            data_dict_vec_numeric = {}
            for key in data_dict_vec:
                new_key = voc_dict[key]
                new_list = [voc_dict[x] for x in data_dict_vec[key]]
                try:
                    data_dict_vec_numeric[new_key] = new_list
                except:
                    data_dict_vec_numeric[new_key] = []
                    data_dict_vec_numeric[new_key] = new_list
                new_key = None
                new_list = None

            Logger.info(Logger().getLogDictInfo('createTrainingData', __name__, 'createTrainingData'),
                        'Populated data_dict_vec_numeric dictionary.', logger)

            # Create pseduo Skip-Gram.
            pseduo_skip_gram_pairs = []
            for key in data_dict_vec_numeric:
                [pseduo_skip_gram_pairs.append([key, x]) for x in data_dict_vec_numeric[key]]

            # Split Pseduo Skip-Gram to sep lists.
            full_data_x = []
            full_data_y = []
            for element in pseduo_skip_gram_pairs:
                full_data_x.append(element[0])
                full_data_y.append([element[1]])

            Logger.info(Logger().getLogDictInfo('createTrainingData', __name__, 'createTrainingData'),
                        'Skip-grams and full data is ready.', logger)

            return full_data_x, full_data_y, voc_dict, voc_dict_reverse, movies

        return createTrainingData

    # Borrowed the data import from keras graph example:
    # https://keras.io/examples/graph/node2vec_movielens/.
    # Implementation skip gram and ultimately learning the
    # vector embeddings is different.
    @_createTrainingData
    def getGraphData(self, min_rating: int = 5) -> tuple:

        Logger.info(Logger().getLogDictInfo(__class__.__name__, __name__, 'getGraphData'),
                    'Retrieving Movie Data.', self.getLogger())

        urlretrieve(
            "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "movielens.zip"
        )
        ZipFile("movielens.zip", "r").extractall()

        Logger.info(Logger().getLogDictInfo(__class__.__name__, __name__, 'getGraphData'),
                    'Retrieved Movie Data.', self.getLogger())

        # Load movies to a DataFrame.
        movies = pd.read_csv("ml-latest-small/movies.csv")

        # Create a `movieId` string.
        movies["movieId"] = movies["movieId"].apply(lambda x: f"movie_{x}")

        # Load ratings to a DataFrame.
        ratings = pd.read_csv("ml-latest-small/ratings.csv")

        # Convert the `ratings` to floating point
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

        # Create the `movie_id` string.
        ratings["movieId"] = ratings["movieId"].apply(lambda x: f"movie_{x}")

        # Filter instances where rating is greater than or equal to min_rating.
        rated_movies = ratings[ratings.rating >= min_rating]

        Logger.info(Logger().getLogDictInfo(__class__.__name__, __name__, 'getGraphData'),
                    'Loaded Movies and Ratings Data. Moving to graph connection compilation.', self.getLogger())

        return rated_movies, movies
