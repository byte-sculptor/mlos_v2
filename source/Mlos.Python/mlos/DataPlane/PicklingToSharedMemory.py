from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process
import pandas as pd
import pickle


from mlos.DataPlane.SharedMemoryDataSet import SharedMemoryDataSet
from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.DataPlane.SharedMemoryDataSetView import SharedMemoryDataSetView


from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store
from mlos.Spaces.HypergridAdapters import CategoricalToDiscreteHypergridAdapter

def make_prediction(model_shared_memory_name, data_set_info: SharedMemoryDataSetInfo):

    shared_memory = SharedMemory(name=model_shared_memory_name, create=False)
    unpickled_model = pickle.loads(shared_memory.buf)

    data_set_view = SharedMemoryDataSetView(data_set_info=data_set_info)
    params_df = data_set_view.get_dataframe()

    pd.set_option('max_columns', None)
    print(params_df)

    prediction = unpickled_model.predict(params_df)
    print(prediction.get_dataframe().describe())

    data_set_view.detach()


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

    shared_memory_data_set = SharedMemoryDataSet(schema=objective_function.parameter_space, shared_memory_name="params2")
    shared_memory_data_set.set_dataframe(df=objective_function.parameter_space.random_dataframe(11))

    worker = Process(target=make_prediction, kwargs={
        'model_shared_memory_name': 'pickled_tree',
        'data_set_info': shared_memory_data_set.get_data_set_info(),
    })
    worker.start()
    worker.join()

    tree_shared_memory.unlink()

    shared_memory_data_set.validate()
    shared_memory_data_set.unlink()

