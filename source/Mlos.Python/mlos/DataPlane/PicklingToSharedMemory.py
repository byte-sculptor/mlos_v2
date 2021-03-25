from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process
from datetime import datetime
import numpy as np
import pandas as pd
import tracemalloc
import time
import pickle

from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store


if __name__ == "__main__":

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config_store.default)
    model_config = decision_tree_config_store.default
    model = DecisionTreeRegressionModel(
        model_config=model_config,
        input_space=objective_function.parameter_space,
        output_space=objective_function.output_space
    )

    params_df = objective_function.parameter_space.random_dataframe(100)
    objectives_df = objective_function.evaluate_dataframe(params_df)
    model.fit(params_df, objectives_df, 0)

    pickled_model = pickle.dumps(model)
    print('breakpoint')

    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=len(pickled_model))
    shm.buf[:] = pickled_model
    unpickled_model = pickle.loads(shm.buf)
    print(unpickled_model.predict(objective_function.parameter_space.random_dataframe(1)).get_dataframe())
    shm.unlink()
