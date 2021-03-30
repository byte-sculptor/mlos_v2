#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import time

from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSetService, SharedMemoryDataSetStoreProxy, SharedMemoryDataSetInfo
from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import ObjectiveFunctionFactory, objective_function_config_store

if __name__ == "__main__":
    service = SharedMemoryDataSetService()
    proxy_connection = service.get_new_proxy_connection()

    objective_function = ObjectiveFunctionFactory.create_objective_function(objective_function_config=objective_function_config_store.default)
    params_df = objective_function.parameter_space.random_dataframe(10)
    try:
        service.launch()


        # Now let's create the proxy and see if we can create a dataset.
        #
        proxy = SharedMemoryDataSetStoreProxy(service_connection=proxy_connection)
        proxy.create_data_set(
            data_set_info=SharedMemoryDataSetInfo(schema=objective_function.parameter_space),
            df=params_df
        )

        time.sleep(1)
    finally:
        service.stop()

    for _, data_set in service._data_set_store._data_sets_by_id.items():
        print(data_set.get_dataframe())
        print(params_df.equals(data_set.get_dataframe()))

