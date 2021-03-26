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
import json


from mlos.DataPlane.SharedMemoryDataSet import SharedMemoryDataSet
from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.DataPlane.SharedMemoryDataSetView import SharedMemoryDataSetView


from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Spaces.HypergridAdapters import CategoricalToDiscreteHypergridAdapter
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonDecoder

def make_prediction(model_shared_memory_name, data_set_info: SharedMemoryDataSetInfo):

    shared_memory = SharedMemory(name=model_shared_memory_name, create=False)
    unpickled_model = pickle.loads(shared_memory.buf)

    schema = json.loads(data_set_info.schema_json_str, cls=HypergridJsonDecoder)
    shared_memory = SharedMemory(name=data_set_info.shared_memory_name, create=False)
    shared_memory_np_records_array = np.recarray(
        shape=data_set_info.shared_memory_np_array_shape,
        dtype=data_set_info.shared_memory_np_array_dtype,
        buf=shared_memory.buf
    )
    df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=schema.dimension_names, index='index')
    shared_memory.close()
    params_df = df

    #params_df = SharedMemoryDataSetView.get_dataframe(data_set_info=data_set_info)

    pd.set_option('max_columns', None)
    print(params_df)

    prediction = unpickled_model.predict(params_df)
    print(prediction.get_dataframe().describe())


if __name__ == "__main__":

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config_store.default)
    model_config = decision_tree_config_store.default
    parameter_space_adapter = CategoricalToDiscreteHypergridAdapter(adaptee=objective_function.parameter_space)
    model = DecisionTreeRegressionModel(
        model_config=model_config,
        input_space=parameter_space_adapter,
        output_space=objective_function.output_space
    )

    params_df = objective_function.parameter_space.random_dataframe(1000)
    projected_params_df = parameter_space_adapter.project_dataframe(params_df, in_place=True)
    objectives_df = objective_function.evaluate_dataframe(projected_params_df)
    model.fit(projected_params_df, objectives_df, 0)

    pickled_model = pickle.dumps(model)
    print(f"pickled model size: {len(pickled_model)}")

    # Let's put the tree in the shared memory.
    #
    tree_shared_memory = SharedMemory(name='pickled_tree', size=len(pickled_model), create=True)
    tree_shared_memory.buf[:] = pickled_model

    # Now let's put some parameters in shared memory.
    #
    #params_df = parameter_space_adapter.project_dataframe(objective_function.parameter_space.random_dataframe(1000000))
    #np_records = params_df.to_records(index=True)
    #params_shared_memory = SharedMemory(name='params_np_array', size=np_records.nbytes, create=True)
    #shared_memory_np_array = np.recarray(shape=np_records.shape, dtype=np_records.dtype, buf=params_shared_memory.buf)
    #np.copyto(dst=shared_memory_np_array, src=np_records)


    shared_memory_data_set = SharedMemoryDataSet(schema=objective_function.parameter_space, shared_memory_name="params2")
    shared_memory_data_set.set_dataframe(df=objective_function.parameter_space.random_dataframe(11))

    worker = Process(target=make_prediction, kwargs={
        'model_shared_memory_name': 'pickled_tree',
        'data_set_info': shared_memory_data_set.get_data_set_info(),
    })
    worker.start()
    worker.join()

    tree_shared_memory.unlink()
    shared_memory_data_set.unlink()

