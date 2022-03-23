from abc import ABC, abstractmethod
import logging
import math
import random
from typing import List

import pandas as pd

from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.Prediction import Prediction
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel
from mlos.Optimizers.RegressionModels.RegressionModelFitState import RegressionModelFitState
from mlos.Spaces import Dimension, Hypergrid, Point, SimpleHypergrid
from mlos.Spaces.HypergridAdapters import HierarchicalToFlatHypergridAdapter
from mlos.Tracer import trace


class EnsembleRegressionModelBase(RegressionModel):
    """ Base class for ensemble models.

    So far we've had the HomogeneousRandomForestRegressionModel as the only ensemble model.
    However, a lot of ideas there are general and can be used by other regression models,
    such as a bootstrapped ensemble lasso cv regression model.

    """

    _PREDICTOR_OUTPUT_COLUMNS = [
        Prediction.LegalColumnNames.IS_VALID_INPUT,
        Prediction.LegalColumnNames.PREDICTED_VALUE,
        Prediction.LegalColumnNames.PREDICTED_VALUE_VARIANCE,
        Prediction.LegalColumnNames.SAMPLE_VARIANCE,
        Prediction.LegalColumnNames.SAMPLE_SIZE,
        Prediction.LegalColumnNames.PREDICTED_VALUE_DEGREES_OF_FREEDOM
    ]

    @trace()
    def __init__(
        self,
        model_type: type,
        model_config: Point,
        input_space: Hypergrid,
        output_space: Hypergrid,
        fit_state: RegressionModelFitState = None,
        logger: logging.Logger = None
    ) -> None:
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        RegressionModel.__init__(
            self,
            model_type=model_type,
            model_config=model_config,
            input_space=input_space,
            output_space=output_space,
            fit_state=fit_state
        )
        assert len(self.target_dimension_names) == 1, "Single target predictions for now."

        self._input_space_adapter = HierarchicalToFlatHypergridAdapter(adaptee=self.input_space)

        self._regressors: List[RegressionModel] = []
        self._create_regressors()
        self._trained = False

    @abstractmethod
    def _create_regressors(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _create_random_flat_subspace(original_space, subspace_name, max_num_dimensions):
        """ Creates a random simple hypergrid from the hypergrid with up to max_num_dimensions dimensions.

        TODO: move this to the *Hypergrid classes.

        :param original_space:
        :return:
        """
        random_point = original_space.random()
        dimensions_for_point = original_space.get_dimensions_for_point(random_point, return_join_dimensions=False)
        selected_dimensions = random.sample(dimensions_for_point, min(len(dimensions_for_point), max_num_dimensions))
        flat_dimensions = []
        for dimension in selected_dimensions:
            flat_dimension = dimension.copy()
            flat_dimension.name = Dimension.flatten_dimension_name(flat_dimension.name)
            flat_dimensions.append(flat_dimension)
        flat_hypergrid = SimpleHypergrid(
            name=subspace_name,
            dimensions=flat_dimensions
        )
        return flat_hypergrid

    @trace()
    def fit(
        self,
        feature_values_pandas_frame: pd.DataFrame,
        target_values_pandas_frame: pd.DataFrame,
        iteration_number: int
    ):
        """ Fits the ensemble model.

            The issue here is that the feature_values will come in as a pandas dataframe  where each column corresponds to one
            of the dimensions in our input space.

            Our goal is to slice them up and feed the observations to individual regressors.

        :param feature_values_pandas_frame:
        :param target_values_pandas_frame:
        :return:
        """
        self.logger.debug(f"Fitting a {self.__class__.__name__} with {len(feature_values_pandas_frame.index)} observations.")

        feature_values_pandas_frame = self._input_space_adapter.project_dataframe(feature_values_pandas_frame, in_place=False)

        for i, regressor in enumerate(self._regressors):
            # Let's filter out samples with missing values
            regressor_input_df = feature_values_pandas_frame[regressor.input_dimension_names]
            non_null_observations = regressor_input_df[regressor_input_df.notnull().all(axis=1)]
            targets_for_non_null_observations = target_values_pandas_frame.loc[non_null_observations.index]

            n_samples_for_regressor = math.ceil(
                min(self.model_config.samples_fraction_per_estimator * len(regressor_input_df.index),
                    len(non_null_observations.index))
            )

            observations_for_regressor_training = non_null_observations.sample(
                n=n_samples_for_regressor,
                replace=False,
                random_state=i,
                axis='index'
            )

            if self.model_config.bootstrap and n_samples_for_regressor < len(regressor_input_df.index):
                bootstrapped_observations_for_regressor_training = observations_for_regressor_training.sample(
                    frac=1.0 / self.model_config.samples_fraction_per_estimator,
                    replace=True,
                    random_state=i,
                    axis='index'
                )
            else:
                bootstrapped_observations_for_regressor_training = observations_for_regressor_training.copy()

            num_selected_observations = len(observations_for_regressor_training.index)
            if regressor.should_fit(num_selected_observations):
                bootstrapped_targets_for_regressor_training = targets_for_non_null_observations.loc[
                    bootstrapped_observations_for_regressor_training.index]
                assert len(bootstrapped_observations_for_regressor_training.index) == len(bootstrapped_targets_for_regressor_training.index)
                regressor.fit(
                    feature_values_pandas_frame=bootstrapped_observations_for_regressor_training,
                    target_values_pandas_frame=bootstrapped_targets_for_regressor_training,
                    iteration_number=len(feature_values_pandas_frame.index)
                )

        self.last_refit_iteration_number = max(tree.last_refit_iteration_number for tree in self._regressors)
        self._trained = any(regressor.trained for regressor in self._regressors)

    @trace()
    def predict(
        self,
        feature_values_pandas_frame: pd.DataFrame,
        include_only_valid_rows: bool =True
    ):
        """ Aggregate predictions from all estimators

        see: https://arxiv.org/pdf/1211.0906.pdf
        section: 4.3.2 for details about how it's done for RandomForest

        :param feature_values_pandas_frame:
        :include_only_valid_rows:
        :return: Prediction
        """
        self.logger.debug(f"Creating predictions for {len(feature_values_pandas_frame.index)} samples.")

        feature_values_pandas_frame = self._input_space_adapter.project_dataframe(feature_values_pandas_frame, in_place=False)

        # dataframe column shortcuts
        is_valid_input_col = Prediction.LegalColumnNames.IS_VALID_INPUT.value
        predicted_value_col = Prediction.LegalColumnNames.PREDICTED_VALUE.value
        predicted_value_var_col = Prediction.LegalColumnNames.PREDICTED_VALUE_VARIANCE.value
        sample_var_col = Prediction.LegalColumnNames.SAMPLE_VARIANCE.value
        sample_size_col = Prediction.LegalColumnNames.SAMPLE_SIZE.value
        dof_col = Prediction.LegalColumnNames.PREDICTED_VALUE_DEGREES_OF_FREEDOM.value

        # collect predictions from ensemble constituent models
        predictions_per_regressor = [
            regressor.predict(feature_values_pandas_frame=feature_values_pandas_frame, include_only_valid_rows=True)
            for regressor in self._regressors
        ]
        prediction_dfs_per_regressor = [prediction.get_dataframe() for prediction in predictions_per_regressor]
        num_prediction_dfs = len(prediction_dfs_per_regressor)

        # We will concatenate all these prediction dataframes together, but to avoid duplicate columns, we first rename them.
        #
        old_names = [predicted_value_col, predicted_value_var_col, sample_var_col, sample_size_col, dof_col]
        predicted_value_col_names_per_regressor = [f"{predicted_value_col}_{i}" for i in range(num_prediction_dfs)]
        mean_var_col_names_per_regressor = [f"{predicted_value_var_col}_{i}" for i in range(num_prediction_dfs)]
        sample_var_col_names_per_regressor = [f"{sample_var_col}_{i}" for i in range(num_prediction_dfs)]
        sample_size_col_names_per_regressor = [f"{sample_size_col}_{i}" for i in range(num_prediction_dfs)]

        for i in range(num_prediction_dfs):
            new_names = [f"{old_name}_{i}" for old_name in old_names]
            old_names_to_new_names_mapping = {old_name: new_name for old_name, new_name in zip(old_names, new_names)}
            prediction_dfs_per_regressor[i].drop(columns=[is_valid_input_col], inplace=True)
            # We can safely overwrite them in place since we are their sole owner by now.
            prediction_dfs_per_regressor[i].rename(columns=old_names_to_new_names_mapping, inplace=True)

        # This creates a 'wide' dataframe with unique column names.
        #
        all_predictions_df = pd.concat(prediction_dfs_per_regressor, axis=1)
        all_predictions_df[predicted_value_col] = all_predictions_df[predicted_value_col_names_per_regressor].apply('mean', axis=1)

        # To compute the pooled variance we will use the second to last form of the equation from the paper:
        #   paper: https://arxiv.org/pdf/1211.0906.pdf
        #   section: section: 4.3.2 for details
        all_predictions_df[predicted_value_var_col] = all_predictions_df[mean_var_col_names_per_regressor].mean(axis=1) \
                                                      + (all_predictions_df[predicted_value_col_names_per_regressor] ** 2).mean(axis=1) \
                                                      - all_predictions_df[predicted_value_col] ** 2 + 0.0000001  # A little numerical instability correction
        all_predictions_df[sample_var_col] = all_predictions_df[sample_var_col_names_per_regressor].mean(axis=1) \
                                             + (all_predictions_df[predicted_value_col_names_per_regressor] ** 2).mean(axis=1) \
                                             - all_predictions_df[predicted_value_col] ** 2 + 0.0000001  # A little numerical instability correction
        all_predictions_df[sample_size_col] = all_predictions_df[predicted_value_col_names_per_regressor].count(axis=1)
        all_predictions_df[dof_col] = all_predictions_df[sample_size_col_names_per_regressor].sum(axis=1) - all_predictions_df[sample_size_col]
        all_predictions_df[is_valid_input_col] = True

        aggregate_predictions = Prediction(
            objective_name=self.target_dimension_names[0],
            predictor_outputs=self._PREDICTOR_OUTPUT_COLUMNS,
            allow_extra_columns=True
        )

        aggregate_predictions_df = all_predictions_df[[column.value for column in self._PREDICTOR_OUTPUT_COLUMNS]]
        aggregate_predictions.set_dataframe(aggregate_predictions_df)
        if not include_only_valid_rows:
            aggregate_predictions.add_invalid_rows_at_missing_indices(desired_index=feature_values_pandas_frame.index)
        return aggregate_predictions
